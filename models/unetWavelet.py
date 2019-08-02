
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
from models.DWT import DWT_Pooling, IWT_UpSampling

def unetWavelet(input_size = (400,400,1)):

    def down_block(input_layer, filters, kernel_size=(3,3), activation="relu"):
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(input_layer)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        return output, DWT_Pooling()(output)


    def up_block(input_layer, residual_layer, filters, kernel_size=(3,3),activation="relu"):
        output = IWT_UpSampling()(input_layer)
        output = Add()([residual_layer,output])
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        output = Conv2D(filters*2, kernel_size, padding="same", activation=activation)(output)
        return output

    inputs = Input(shape = input_size)

    down1, pool1 = down_block(inputs,64)
    down2, pool2 = down_block(pool1,128)
    down3, pool3 = down_block(pool2,256)
    down4, pool4 = down_block(pool3,512)

    down5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation ="relu")(pool4)
    down5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation ="relu")(down5)
    down5 = Conv2D(filters=2048, kernel_size=(3, 3), padding="same", activation ="relu")(down5)

    up = up_block(down5,down4,512)
    up = up_block(up,down3,256)
    up = up_block(up,down2,128)
    up = up_block(up,down1,64)

    output = Conv2D(filters=input_size[2], kernel_size=(1, 1), padding="same")(up)
    model = Model(input = inputs, output = output)
    
    return model