def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False


def parse_args(parser):
    # model
    parser.add_argument('--loss-reduction', default='sum', choices=['mean', 'sum'])
    ## encoder
    parser.add_argument('--model-encoder-channels', default=64, type=int)
    parser.add_argument('--model-encoder-pe-dim', default=20, type=int)
    parser.add_argument('--model-encoder-num-layers', default=3, type=int)
    parser.add_argument('--model-encoder-dropout', default=0.5, type=float)
    ## decoder
    parser.add_argument('--model-decoder-channels', default=64, type=int)
    parser.add_argument('--model-decoder-num-layers', default=2, type=int)
    parser.add_argument('--model-decoder-dropout', default=0.1, type=float)
    # training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr-factor', default=0.9, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--train-batch-size', default=512, type=int)
    parser.add_argument('--more-tasks', default=False, type=str2bool)
    parser.add_argument('--smi-leakage-method', default='none', choices=['none', 'test+valid', 'test'])
    # evaluation
    parser.add_argument('--eval-methods', default=['improvement', 'pps', 'last', 'independent'], nargs='+', type=str)
    # misc
    parser.add_argument('--wandb', action='store_true', help='use wandb')

    args = parser.parse_args()
    assert len(args.eval_methods) > 0, 'No evaluation method provided'
    print(args.eval_methods)
    return args
