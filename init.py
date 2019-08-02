import argparse, os
from train import train
from test import test

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-tr', '--train-path',
                        type=str,
                        default=os.getcwd() + 'DIV2K/DIV2K_train_HR/',
                        help='The path to the training dataset')


    parser.add_argument('-t', '--test-path',
                        type=str,
                        default=os.getcwd() + 'DIV2K/DIV2K_valid_HR/',
                        help='The path to the test dataset')

    parser.add_argument('-n', '--noise',
                        type=int,
                        default=15,
                        help='Noise level')

    parser.add_argument('-s', '--sliding_window',
                        type=int,
                        default=0,
                        help='Use sliding window for testing to avoid grid effect.')

    parser.add_argument('-lw', '--load-weights-path',
                        type=str,
                        default=os.getcwd(),
                        help='Path to weights file to load')

    parser.add_argument('-a', '--architecture',
                    choices=['unet', 'wavelet'],
                    default='unet',
                    help='Select architecture')

    parser.add_argument('-m', '--mode',
                        choices=['train', 'test'],
                        default='train',
                        help='Train or test a model')

    FLAGS, unparsed = parser.parse_known_args()


    if FLAGS.mode.lower() == 'train':
        print("train")
        train(FLAGS)

    elif FLAGS.mode.lower() == 'test':
        print("test")
        test(FLAGS)

    else:
        raise RuntimeError('Unkwown mode, please choose from "train" or "test".')
