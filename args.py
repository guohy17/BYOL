import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='STL10')
    parser.add_argument('--mlp_size', type=int, default=2048)
    parser.add_argument('--pro_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--m', type=float, default=0.996)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--weight_decay', type=int, default=0.0004)
    args = parser.parse_args()
    return args