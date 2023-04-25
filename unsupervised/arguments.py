import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='RD-GCL')
    parser.add_argument('--DS', dest='DS', type=str, default='NCI1', help='Dataset')

    parser.add_argument('--lr', dest='lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
                        help='Number of graph convolution layers before each pooling')

    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32, help='hidden dimension')

    parser.add_argument('--aug', type=str, default='subgraph')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--log_interval', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--global_size', type=float, default=0.8)
    parser.add_argument('--local_size', type=float, default=0.2)
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--log', type=str, default='full')

    parser.add_argument('--step-size', type=float, default=4e-3)
    parser.add_argument('--delta', type=float, default=8e-3)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--pp', type=str, default="X",
                        help='perturb_position (default: X(feature), H(hidden layer))')

    return parser.parse_args()

