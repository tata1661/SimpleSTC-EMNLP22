import argparse
import torch
from Trainer import Trainer
import time
from utils.utils import set_seed

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description = "model params")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", "-d", type=str, default='snippets')
    parser.add_argument("--train_num", type=int, default=100)
    parser.add_argument("--file_dir", "-f_dir", type=str, default='./')
    parser.add_argument("--data_path", "-d_path", type=str, default='./data/')
    parser.add_argument('--disable_cuda', action='store_true')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=200) 
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--drop_out", type=float, default=0.9)
    parser.add_argument("--max_epoch", type=int, default=1000)
    params = parser.parse_args()

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    set_seed(params.seed)
    trainer = Trainer(params)
    test_acc,best_f1 = trainer.train()
    del trainer
    print('total time: ', time.time() - start)