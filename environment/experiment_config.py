
def add_environment_args(parser):
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--num-distractors', type=int, default=3)
    parser.add_argument('--view-size', type=int, default=7)


def environment_kwargs(args):
    return dict(size=args.size, num_distractors=args.num_distractors,
                agent_view_size=args.view_size)
