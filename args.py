import argparse

parser = argparse.ArgumentParser()
#Path setting
parser.add_argument(
            "--data_dir",
            type=str,
            default="./data/H36M",
            help="path to H36M dataset",
        )
parser.add_argument(
    "--data_dir_3dpw",
    type=str,
    default="./data/3DPW/sequenceFiles",
    help="path to 3DPW dataset",
)
parser.add_argument(
    "--data_dir_cmu",
    type=str,
    default="./data/CMU",
    help="path to CMU dataset",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="./save_model",
    help="path to pretrained model",
)
#Loading setting
parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument("--dataset", type=str, choices=["H36M", "CMU", "3DPW"], default="H36M")
parser.add_argument(
            "--actions", type=str, default="all", help="which activities are used"
        )#"all""debug""all_srnn"
parser.add_argument(
            "--is_load",
            dest="is_load",
            action="store_true",
            help="wether to load existing model",
        )
parser.add_argument(
            "--sample_rate", type=int, default=2, help="frame sampling rate"
        )
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument(
            "--job", type=int, default=20, help="subprocesses to use for data loading"
        )
#Model setting
parser.add_argument(
            "--dropout",
            type=float,
            default=0.25,
            help="dropout probability, 1.0 to make no dropout",
        )
parser.add_argument('--input_len', default=10, type=int, help="the length of the input sequence")
parser.add_argument('--seq_len', default=20, type=int, help="the length of the entire sequence(input and output)")
parser.add_argument('--joints_input', default=22, type=int)#h36m-22, cmu-25, 3DPW-24
parser.add_argument('--lr', default=5e-4, help="the learning rate")
parser.add_argument('--step_size', default=2, help="the step size for the learning rate decay")
parser.add_argument('--gamma', default=0.97, help="the rate of the learning rate decay")
parser.add_argument('--interval', default=5000)
parser.add_argument("--train_batch", type=int, default=64)#h36m-64, cmu-16 3DPW-16
parser.add_argument("--test_batch", type=int, default=128)
parser.add_argument('--joints_total', default=32, type=int)
#Attack setting
parser.add_argument('--epsilon', default=1e-2, type=float, help="The boundary of the perturbations")
parser.add_argument('--epsilon_step', default=1e-3, type=float, help="The step size of the attack optimization in each iteration")
parser.add_argument('--iters', default=50, type=int, help="The number of iterations the attack optimization needs")
parser.add_argument('--attack_range', nargs='+', default=[], type=int, help="The frames need to be perturbed(if empty then attack all the frames)")

