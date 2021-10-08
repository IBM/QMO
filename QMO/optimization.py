import logging
import os
from functools import partial
from time import time

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from QMO.losses import (QED, loss_function, morgan_fingerprint, penalized_logP,
                        tanimoto_similarity)

LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

def write_log(writer, z, grad, z_0, mol, smiles, fp_0, step):
    pen_logP = penalized_logP(mol)
    qed = QED(mol)
    sim = tanimoto_similarity(mol, fp_0)

    writer.add_histogram('z', z, step)
    writer.add_histogram('grad', grad, step)
    writer.add_scalar('zdist', np.linalg.norm(z.cpu() - z_0), step)
    writer.add_scalar('pen_logP', pen_logP, step)
    writer.add_scalar('qed', qed, step)
    writer.add_scalar('sim', sim, step)

    return dict(pen_logP=pen_logP, qed=qed, sim=sim)

def estimate_gradient(z, q, beta, criterion, sigma=100):
    z_dim = z.shape[1]
    u = np.random.normal(0, sigma, size=(q, z_dim)).astype('float32')
    u = torch.from_numpy(u / np.linalg.norm(u, axis=1, keepdims=True))

    f_0 = criterion(z)
    f_tmp = criterion(z + beta*u)
    return torch.mean(z_dim * u * np.expand_dims(f_tmp - f_0, 1)/ beta, dim=0,
                      keepdims=True).to(z.dtype)

def optimize(model, seq, q=10, base_lr=0.1, max_iter=1000, num_restarts=1,
             weight=1, beta=1, use_adam=True, early_stop=False, score=None,
             constraints=[], writer=None, run_str=None, results_dir='results',
             init_best={}, write_log=write_log, flip_weight=False):
    z_0 = model.encode(seq)
    fp_0 = morgan_fingerprint(Chem.MolFromSmiles(seq))

    LOG.info(f'Original sequence: {seq}')
    LOG.info(f"Reconst. sequence: {model.decode(z_0)[1][0]}")

    loss = partial(loss_function, model=model, weight=weight, score=score,
                   constraints=constraints, weight_constraint=flip_weight)
    best = {'score': -np.inf, 'found': False, 'early_stop': False}
    best.update(init_best)
    start_time = time()
    for k in range(num_restarts):
        if best['early_stop']:
            break

        z = z_0.clone()
        traj_z, traj_loss = [z.clone().numpy()], [loss(z)]
        adam = torch.optim.Adam([z], lr=base_lr)
        for i in tqdm(range(max_iter)):
            grad = estimate_gradient(z, q, beta, loss)
            if use_adam:
                z.grad = grad
                adam.step()
            else:
                lr = ((1 - i/max_iter)**0.5) * base_lr
                z -= grad * lr
            z.clamp_(-1, 1)

            traj_z.append(z.clone().numpy())
            traj_loss.append(loss(z))

            mol, sample = model.decode(z)
            mol_score = score and -score(mol, sample)[0]
            desc = write_log(writer, z, grad, z_0, mol[0], sample[0], fp_0, i)
            if (score is None or mol_score > best['score']) and all(c(mol, sample) == 0 for c in constraints):
                best.update(desc)
                best.update(dict(step=i, z=z, z_0=z_0, seq=sample[0],
                                 score=mol_score, found=True, run=k,
                                 time=time()-start_time, early_stop=early_stop,
                                 decode_failures=model.decode_failure_count,
                                 total_decodes=model.total_decoding))

                LOG.info(f'PASSED! {desc}')
                np.savez(os.path.join(results_dir, run_str), **best)

                if early_stop:
                    break
        np.savez(os.path.join(results_dir, f"TRAJ{k}_"+run_str),
                    z=np.stack(traj_z), loss=np.array(traj_loss))
    if not best['found']:
        LOG.info('Search failed!')
    return best
