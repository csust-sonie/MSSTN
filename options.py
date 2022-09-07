import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=400, help='train epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')

parser.add_argument('--dataset', type=str,
                    default='NPEMS04', help="dataset select")
parser.add_argument('--train_rate', type=float, default=0.7,
                    help='the ratio of training dataset')
parser.add_argument('--val_rate', type=float, default=0.15,
                    help='the ratio of validating dataset')
parser.add_argument('--test_rate', type=float, default=0.15,
                    help='the ratio of validating dataset')

parser.add_argument('--n_history', type=int, default=12,
                    help='the length of history time series of input')
parser.add_argument('--n_predict', type=int, default=12,
                    help='the length of target time series of for prediction')


parser.add_argument('--sigma', type=float, default=0.1,
                    help='sigma fro the spatial matrix')
parser.add_argument('--thres', type=float, default=0.6,
                    help='the threshold for the spatial matrix')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')

parser.add_argument('--log', action='store_true', help='if write log to file')
parser.add_argument('--tensorboard', action='store_true',
                    help='if write loss to the tensorboard')

parser.add_argument('--cuda_visiable', type=str,
                    default='0', help="gpu visiable")
parser.add_argument('--sample', type=str, default='normal',
                    help='if sample function is normal, then the data of source not with the preweek in the same time segment')

args = parser.parse_args()


if __name__ == "__main__":
    from loguru import logger

    if args.log:
        logger.add("log_{time}.log")

    options = vars(args)

    if args.log:
        logger.info(options)
    else:
        print(options)
