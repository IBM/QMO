from abc import ABC, abstractmethod
import os.path as osp

from cddd.inference import InferenceModel
import numpy as np
from rdkit import Chem
import torch

class Model(ABC):
    """Basic encoder-decoder model abstract base class.

    Attributes:
        decode_failure_count (float): Running tally of invalid decoded strings.
        total_decoding (float): Running tally of total decodes.
    """
    def __init__(self):
        self.decode_failure_count = 0
        self.total_decoding = 0

    @abstractmethod
    def decode(self, z):
        """Decode latent vectors.

        Args:
            z (torch.Tensor): Batch of latent vectors.

        Returns:
            tuple:
                * **mols** (`list`) -- List of `:class:~rdkit.Chem.rdchem.Mol`.
                * ** samples** (`list`) -- List of SMILES strings.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, seq):
        """Encode sequences.

        Args:
            seq (list): List of (string) sequences.

        Returns:
            torch.Tensor: Latent representation.
        """
        raise NotImplementedError

class CDDDModel(Model):
    def __init__(self, model_dir='../models/default_model', eps_clip=1e-7):
        """Load CDDD model.

        Args:
            model_dir (str): Path to model (e.g. ``.../default_model``).
        """
        super().__init__()
        # make path relative to this file if not already absolute
        if not osp.isabs(model_dir):
            model_dir = osp.join(osp.dirname(osp.abspath(__file__)), model_dir)
        self.model = InferenceModel(model_dir=model_dir, num_top=2)
        self.eps_clip = eps_clip

    def decode(self, z):
        z = torch.clamp(z, -1, 1)
        num_embs = 1 if z.ndim == 1 else z.shape[0]
        self.total_decoding += num_embs
        samples = self.model.emb_to_seq(z.numpy())  # [n x 2]
        if isinstance(samples[0], str):
            samples = [samples]
        mols, samples1 = [], []
        for s0, s1 in samples:
            mol0 = Chem.MolFromSmiles(s0)
            if mol0 is not None:
                mols.append(mol0)
                samples1.append(s0)
                continue
            mol1 = Chem.MolFromSmiles(s1)
            if mol1 is not None:
                mols.append(mol1)
                samples1.append(s1)
            else:
                self.decode_failure_count += 1
                mols.append(None)
                samples1.append(s0)
        return mols, samples1

    def encode(self, seq):
        orig_emb = self.model.seq_to_emb(seq)
        z_0 = torch.tensor(orig_emb).detach().cpu()
        return z_0