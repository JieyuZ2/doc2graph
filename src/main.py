import sys
sys.path.append('./')
import argparse
import os.path as osp
import networkx as nx
from datetime import timedelta
import matplotlib.pyplot as plt

from src.utils import *
from src.models import *
from src.dataset import Dataset, SST_Dataset, DBLP_Dataset, NYnews_Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    # general options
    parser.add_argument('--dataset', type=str, default='sst', choices=['sst', 'dblp', 'nyt'])
    parser.add_argument("--embed_path", type=str, default='')
    parser.add_argument('--model', type=str, default='netgen', choices=['netgen', 'rnnvae', 'rnnvae2', 'rnnae'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--prefix", type=str, default='', help="prefix use as addition directory")
    parser.add_argument('--suffix', default='', type=str, help='suffix append to log dir')
    parser.add_argument('--log_level', default=20)
    parser.add_argument('--log_every', type=int, default=1, help='log results every epoch.')
    parser.add_argument('--save_every', type=int, default=10, help='save learned embedding every epoch.')

    # training options
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=0)
    parser.add_argument('--minimal_epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=100)

    # evaluation options
    parser.add_argument('--eval_epochs', type=int, default=1000)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--eval_lr', type=float, default=1e-3)
    parser.add_argument('--eval_early_stop', type=int, default=1)
    parser.add_argument('--eval_minimal_epoch', type=int, default=10)
    parser.add_argument('--eval_patience', type=int, default=50)


    # model options
    parser.add_argument('--h_dim', type=int, default=50)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--n_node', type=int, default=5)
    parser.add_argument('--kl_weight', type=float, default=0.1)
    parser.add_argument('--cls_weight', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.01)

    return parser.parse_args()


def save_model(model, logger, path, name='NetGen.bin'):
    if not os.path.exists(path):
        os.makedirs(path)
    path = osp.join(path, name)
    logger.info(f'saving model to {path}')
    torch.save(model.state_dict(), path)


def plot_graph(adj, words, path):
    G = nx.Graph()
    G.add_nodes_from(words)
    n_node = len(adj)
    for i in range(n_node):
        for j in range(i):
            if adj[i][j] > 0.5:
                G.add_edge(words[i], words[j])
    nx.draw(G, with_labels=True)
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()


def main(args):
    start_time = time.time()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset_map = {
        'sst': SST_Dataset,
        'dblp': DBLP_Dataset,
        'nyt': NYnews_Dataset
    }
    dataset = dataset_map[args.dataset]()
    args.n_labels = dataset.n_labels

    logger = init_logger(args)
    print_config(args, logger)

    if args.model == 'rnnvae':
        model = RNNVAE(dataset.n_vocab, args.n_labels, args.h_dim, args.z_dim,
                       pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True, device=device)
    elif args.model == 'rnnvae2':
        model = RNNVAE2(dataset.n_vocab, args.n_labels, args.h_dim, args.z_dim,
                       pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True, device=device)
    elif args.model == 'rnnae':
        model = RNNAE(dataset.n_vocab, args.n_labels, args.h_dim, args.z_dim,
                        pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=True, device=device)
    elif args.model == 'netgen':
        model = NetGen(args.n_node, dataset.n_vocab, args.n_labels, args.h_dim, args.z_dim,
                       pretrained_embeddings=dataset.get_vocab_vectors(), freeze_embeddings=False, device=device)
    else:
        raise NotImplementedError(f'the model {args.model} is not implemented!')

    try:
        model.train_model(args, dataset)
    except KeyboardInterrupt:
        save_model(model, logger, args.log_dir)
        exit('KeyboardInterrupt!')

    logger.info("total cost time: {} ".format(timedelta(seconds=(time.time() - start_time))))
    save_model(model, logger, args.log_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
