import argparse


def general_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="a dummy argument to fool ipython", default="1")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max-causal-epochs', type=int, default=200,
                        help='Maximal number of epochs to train every time RL adds a new datapoint.')
    parser.add_argument('--train-bs', type=int, default=10,
                        help='Number of samples per batch during training.')
    parser.add_argument('--val-bs', type=int, default=10,
                        help='Number of samples per batch during validation and test.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--decoder-hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature for Gumbel softmax.')
    parser.add_argument('--input-atoms', type=int, default=6,
                        help='Number of atoms need to be controlled in simulation.')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Suffix for training data (e.g. "_charged".')
    parser.add_argument('--val-suffix', type=str, default=None,
                        help='Suffix for valid and testing data (e.g. "_charged".')
    parser.add_argument('--decoder-dropout', type=float, default=0.0,
                        help='Probability of an element to be zeroed.')
    parser.add_argument('--save-folder', type=str, default='logs_RL',
                        help='Where to save the trained model and logs.')
    parser.add_argument('--edge-types', type=int, default=2,
                        help='The number of edge types to infer.')
    parser.add_argument('--dims', type=int, default=9,
                        help='The number of input dimensions (position + velocity).')
    parser.add_argument('--timesteps', type=int, default=40,
                        help='The number of time steps per sample.')
    parser.add_argument('--prediction-steps', type=int, default=20, metavar='N',
                        help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--lr-decay', type=int, default=40,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--skip-first', action='store_true', default=True,
                        help='Skip first edge type in decoder, i.e. it represents no-edge.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--self-loop', action='store_true', default=True,
                        help='Whether graph contains self loop.')
    parser.add_argument('--kl', type=float, default=10,
                        help='Whether to include kl as loss.')
    parser.add_argument('--action_dim', type=int, default=4,
                        help='Dimension of action.')

    parser.add_argument('--val-variations', type=int, default=4,
                        help='#values for one controlled var in validation dataset.')
    parser.add_argument('--target-atoms', type=int, default=2,
                        help='#atoms for results.')
    parser.add_argument('--comment', type=str, default='',
                        help='Additional info for the run.')
    parser.add_argument('--train-size', type=int, default=None,
                        help='#datapoints for train')
    parser.add_argument('--val-size', type=int, default=None,
                        help='#datapoints for val')
    parser.add_argument('--test-size', type=int, default=None,
                        help='#datapoints for test')
    parser.add_argument('--val-need-grouping', action='store_true', default=False,
                        help='If grouped is True, whether the validation dataset actually needs grouping.')
    parser.add_argument('--val-grouped', action='store_true', default=False,
                        help='Whether to group the valid and test dataset')
    parser.add_argument('--control-constraint', type=float, default=0.0,
                        help='Coefficient for control constraint loss')
    parser.add_argument('--gt-A', action='store_true', default=False,
                        help='Whether use the ground truth adjacency matrix, useful for debuging the encoder.')
    parser.add_argument('--train-log-freq', type=int, default=10,
                        help='How many epochs every logging for causal model training.')
    parser.add_argument('--val-log-freq', type=int, default=5,
                        help='How many epochs every logging for causal model validating.')
    parser.add_argument('--all-connect', action='store_true', default=False,
                        help='Whether the adjancency matrix is fully connected and not trainable.')
    parser.add_argument('--intervene-strength', type=int, default=1,
                        help='How much confidence we add to the relation graph every time a noisy intervention is performed.')

    return parser
