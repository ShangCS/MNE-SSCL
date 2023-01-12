import argparse
from models import MNESSCL
import warnings

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hid_units', type=int, default=128, help='hidden dimension')
    parser.add_argument('--dataset', nargs='?', default='acm')
    parser.add_argument('--sc', type=float, default=0.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')

    parser.add_argument('--nb_epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=0, help='the id of gpu to use')
    parser.add_argument('--k', type=int, default=5, help='Clustering number parameter')
    parser.add_argument('--t', type=float, default=1, help='Sim function parameter')
    
    parser.add_argument('--coef_n', type=float, default=1.0, 
                        help='Intra_view structure leve: coefficient for the structure loss of intra-view')
    parser.add_argument('--coef_n_v', type=float, default=1.0,
                        help='Inter_view structure leve: coefficient for the node structure of inter-view')
    parser.add_argument('--coef_c', type=float, default=1.0,
                        help='Intra_view semantic leve: coefficient for the semantic loss of intra-view')
    parser.add_argument('--coef_c_v', type=float, default=1.0,
                        help='Inter_view semantic leve: coefficient for the semantic loss of inter-view')
    
    parser.add_argument('--save_folder', type=str, default="./Results", help='folder for saving the model')

    return parser.parse_known_args()

def printConfig(args):
    arg2value = {}
    print("Parameters:")
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

def main():
    args, unknown = parse_args()
    printConfig(args)

    embedder = MNESSCL(args)
    macro_f1s, micro_f1s, nmi, sim = embedder.training()
    
    return macro_f1s, micro_f1s, nmi, sim

if __name__ == '__main__':
    macro_f1s, micro_f1s, nmi, sim = main()
