import argparse
import sys

from VAE_trainer import Trainer


def parse_args(args):
    parser = argparse.ArgumentParser();

    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--numEpochs', default=10, type=int)

    return parser.parse_args(args)



if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)

    trainer = Trainer(args)
    trainer.train()

    