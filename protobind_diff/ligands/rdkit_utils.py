import importlib
from typing import Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from rdkit import Chem

from FPSim2 import FPSim2Engine
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, Descriptors
from rdkit.DataStructs import BulkTanimotoSimilarity
from sklearn.cluster import DBSCAN
import scipy
RDLogger.DisableLog('rdApp.*')


class BoostWrapper(object):
    """ Help joblib to deal with boost functions """
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module = importlib.import_module(module_name)

    @property
    def method(self):
        return getattr(self.module, self.method_name)

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


def cluster_fpsim2(distance_path, smiles_h5_path=None, dist_eps=0.15):
    """ Cluster precomputed FPSim2 distance matrix using DBSCAN algorithm """
    if isinstance(distance_path, str):
        distance_path = Path(distance_path, smiles_h5_path=None)

    if smiles_h5_path is None:
        smiles_h5_path = distance_path.parent / 'all_smiles.h5'
    precomputed_indices = FPSim2Engine(smiles_h5_path).fps[:, 0]
    map_precomputed = np.argsort(precomputed_indices)  # maps original smiles order to FPSim2 order

    precomputed_distance = scipy.sparse.load_npz(distance_path)
    db = DBSCAN(eps=dist_eps, min_samples=1, metric='precomputed', n_jobs=-1)
    labels = db.fit_predict(precomputed_distance)

    # df_ = pd.DataFrame(data=smiles.keys(), index=list(smiles.values()), columns=['SMILES'])
    # df_ = df_.sort_index()
    # df_['cluster'] = labels[map_precomputed]
    return labels[map_precomputed]


def tanimoto_smiles(mol1, mol2, fp='rdkit', bits=2048, radius=2):

    if isinstance(mol1, str):
        mol1 = Chem.MolFromSmiles(mol1)
    if isinstance(mol2, str):
        mol2 = Chem.MolFromSmiles(mol2)

    _supported_fps = {
        'rdkit': Chem.RDKFingerprint,
        'morgan': Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect,
        'maccs': Chem.rdMolDescriptors.GetMACCSKeysFingerprint,
    }
    if fp not in _supported_fps:
        raise ValueError(f"Fingerprint {fp} is not supported, available fps {_supported_fps.keys()}")

    ffp = None
    if fp == 'rdkit':
        ffp = lambda x: _supported_fps[fp](x, fpSize=bits)
    elif fp == 'morgan':
        ffp = lambda x: _supported_fps[fp](x, fpSize=bits, radius=radius, nBits=bits)
    elif fp == 'maccs':
        ffp = _supported_fps[fp]

    return rdkit.DataStructs.TanimotoSimilarity(ffp(mol1), ffp(mol2))


def validate_smile(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        Chem.SanitizeMol(mol)
        return smile
    except Exception:
        return None


def calc_chem_desc(smiles):
    rdkit_features = {'MolWt': rdkit.Chem.Descriptors.MolWt,
                      'MolLogP': rdkit.Chem.Descriptors.MolLogP,
                      'NumRotatableBonds': rdkit.Chem.Descriptors.NumRotatableBonds,
                      'CalcTPSA': rdkit.Chem.rdMolDescriptors.CalcTPSA,
                      'RingCount': rdkit.Chem.Descriptors.RingCount,
                      }
    if isinstance(smiles[0], str):
        mols = smiles_to_mols(smiles)
    elif isinstance(smiles[0], rdkit.Chem.rdchem.Mol):
        mols = smiles
    else:
        raise TypeError(f'smiles must be a string or a rdkit.Chem.rdchem.Mol: {type(smiles[0])}')
    res = {}
    for name, func in rdkit_features.items():
        res[name] = np.asarray([func(m) if m is not None else np.nan for m in mols ])
    return pd.DataFrame(res)


def smiles_to_mols(smiles, n_jobs=8):
    if isinstance(smiles, (list, tuple, np.ndarray)):
        pass
    elif isinstance(smiles, pd.Series):
        smiles = smiles.tolist()
    else:
        raise TypeError(f"{type(smiles)=}")

    assert len(smiles) > 0
    assert isinstance(smiles[0], str), f"expect smiles string, got f{smiles[0]}"

    mols = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(BoostWrapper('MolFromSmiles', 'rdkit.Chem.rdmolfiles', ))(smi) for smi in smiles)
    return mols


def smiles_to_fps(smiles_or_mols, finger_type='rdkit', n_jobs=8, fp_param=None):
    if isinstance(smiles_or_mols, (list, tuple, np.ndarray)):
        pass
    elif isinstance(smiles_or_mols, pd.Series):
        smiles_or_mols = smiles_or_mols.tolist()
    else:
        raise TypeError(f"{type(smiles_or_mols)=}")

    assert len(smiles_or_mols) > 0
    assert isinstance(smiles_or_mols[0],
                      (str, rdkit.Chem.rdchem.Mol)), f"variable {smiles_or_mols[0]} has type {type(smiles_or_mols[0])}"

    if isinstance(smiles_or_mols[0], str):
        mols = smiles_to_mols(smiles_or_mols)
    else:
        mols = smiles_or_mols

    if fp_param is None:
        fp_param = {}
    fp_func, fp_func_name, fp_func_module, fp_params = _find_fingerprint_function(finger_type)
    fp_params.update(fp_param)
    if finger_type == 'morgan':
        fp_func = fp_func(**fp_params).GetFingerprint
        fp_params = {}
    fps = joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(fp_func)(mol, **fp_params) for mol in mols)
    return fps


def _find_fingerprint_function(finger_type: str) -> Tuple[callable, str, str, dict]:
    kwargs = {}
    if finger_type == 'rdkit':
        fp_func_name = 'RDKFingerprint'
        fp_func_module = 'rdkit.Chem'
    elif finger_type == 'maccs':
        fp_func_name = 'GetMACCSKeysFingerprint'
        fp_func_module = 'rdkit.Chem.rdMolDescriptors'
    elif finger_type == 'morgan':
        fp_func_name = 'GetMorganGenerator'
        fp_func_module = 'rdkit.Chem.AllChem'
        kwargs = dict(atomInvariantsGenerator=rdkit.Chem.rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
                      radius=2, fpSize=2048, countSimulation=True)
    else:
        raise NotImplementedError(f"Use `rdkit` or `maccs` or `morgan` as fps")

    fp_func = getattr(importlib.import_module(fp_func_module), fp_func_name)
    return fp_func, fp_func_name, fp_func_module, kwargs


def randomize_smiles_rotated(smiles: str, with_order_reversal: bool = True) -> str:
    """
    Randomize a SMILES string by doing a cyclic rotation of the atomic indices.

    Adapted from https://github.com/GLambard/SMILES-X/blob/758478663030580a363a9ee61c11f6d6448e18a1/SMILESX/augm.py#L19.

    The outputs of this function can be reproduced by setting the seed with random.seed().

    Raises:
        InvalidSmiles: for invalid molecules.

    Args:
        smiles: SMILES string to randomize.
        with_order_reversal: whether to reverse the atom order with 50% chance.

    Returns:
        Randomized SMILES string.
    """

    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    n_atoms = mol.GetNumAtoms()

    # Generate random values
    rotation_index = np.random.randint(0, n_atoms - 1)
    reverse_order = with_order_reversal and np.random.choice([True, False])

    # Generate new atom indices order
    atoms = list(range(n_atoms))
    new_atoms_order = (
        atoms[rotation_index % len(atoms) :] + atoms[: rotation_index % len(atoms)]
    )
    if reverse_order:
        new_atoms_order.reverse()

    mol = Chem.RenumberAtoms(mol, new_atoms_order)
    return Chem.MolToSmiles(mol, canonical=False)