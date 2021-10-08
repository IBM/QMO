import argparse
from functools import partial
import os

from rdkit import Chem, RDLogger
import numpy as np
from tensorboardX import SummaryWriter
import tensorflow as tf
import torch

from QMO.models import CDDDModel
from QMO.losses import loss_tanimoto, loss_pen_logP_imp, loss_QED
from QMO.losses import morgan_fingerprint, penalized_logP, QED
from QMO.optimization import optimize


# Suppress warnings
tf.logging.set_verbosity(tf.logging.ERROR)
RDLogger.logger().setLevel(RDLogger.CRITICAL)

def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        torch.manual_seed(args.seed)

    model = CDDDModel()
    # canonicalize
    mol_0 = Chem.MolFromSmiles(args.seq)
    seq = Chem.MolToSmiles(mol_0, isomericSmiles=False)

    run_str = '~'.join(f'{k}={v}' for k, v in vars(args).items() if k != 'seq')
    # FIXME: using SMILES in path may cause issues...
    results_dir = os.path.join('results', args.seq)
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join('runs', args.seq, run_str))

    if args.score == 'qed':
        score = partial(loss_QED, penalty=QED(mol_0)-0.1)
    elif args.score == 'logP':
        score = partial(loss_pen_logP_imp, pen_logP_0=penalized_logP(mol_0))
    elif args.score == 'sim':
        score = partial(loss_tanimoto, fp_0=morgan_fingerprint(mol_0))
    else:
        score = None

    constraints = []
    if args.qed is not None:
        constraints.append(partial(loss_QED, threshold=args.qed))
    if args.logP is not None:
        constraints.append(partial(loss_pen_logP_imp,
                pen_logP_0=penalized_logP(mol_0), threshold=args.logP))
    if args.sim is not None:
        constraints.append(partial(loss_tanimoto,
                fp_0=morgan_fingerprint(mol_0), threshold=args.sim))

    if isinstance(args.early_stop, float):
        init_best = {'score': args.early_stop if args.score != 'tox' else
                             -args.early_stop}
        args.early_stop = True
    else:
        init_best = dict()

    optimize(model, seq, q=args.num_grad_samples, base_lr=args.base_lr,
             max_iter=args.max_iter, num_restarts=args.num_restarts,
             weight=args.weight, beta=args.beta, use_adam=args.adam,
             early_stop=args.early_stop, score=score, constraints=constraints,
             writer=writer, run_str=run_str, results_dir=results_dir,
             init_best=init_best, flip_weight=args.flip_weight)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--max_iter', default=20, type=int,
            help='Max iters to find a solution that satisfies the threshold.')
    parser.add_argument('-k', '--num_restarts', default=50, type=int,
            help='Number of restarts.')
    parser.add_argument('-w', '--weight', default=0.25, type=float,
            help='Weight multiplying the property loss in the total loss.')
    parser.add_argument('--flip-weight', action='store_true', default=False,
            help='Apply weight to constraint instead of property in loss '
            'function.')
    parser.add_argument('--beta', default=10, type=float,
            help='Smoothing parameter.')
    parser.add_argument('-q', '--num_grad_samples', default=50, type=int,
            help='Number of gradient estimation samples (Q).')
    parser.add_argument('--base_lr', default=0.2, type=float,
            help='Starting learning rate.')
    parser.add_argument('--adam', action='store_true', default=False,
            help='Use Adam instead of inverse sqrt decay.')
    parser.add_argument('--early-stop', nargs='?', default=False, const=True,
            type=float, help='Stop after first success. If no arguments are '
            'provided, stop if all constrained thresholds are satisfied. If a '
            'float argument is provided, this will be used as a threshold for '
            'the score loss in addition to the constraints.')
    parser.add_argument('-s', '--seed', type=int, default=None,
            help='Random seed.')

    parser.add_argument('--score', type=str, choices=['qed', 'logP', 'sim'],
            help='Metric to maximize as a score.')

    constraints = parser.add_argument_group('Constrained losses')
    constraints.add_argument('--qed', type=float, default=None,
            help='QED threshold.')
    constraints.add_argument('--logP', type=float, default=None,
            help='Penalized logP threshold.')
    constraints.add_argument('--sim', type=float, default=None,
            help='Similarity threshold.')

    parser.add_argument('seq', type=str, help='Sequence to optimize.')

    args = parser.parse_args()
    main(args)
