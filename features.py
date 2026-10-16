import numpy as np
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
 
 
def aromatic_proportion(molecule):
    num_aromatic = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic())
    return num_aromatic / molecule.GetNumHeavyAtoms()
 
descriptor_functions = {                                                #delaney equation descriptors
    "MolLogP": Descriptors.MolLogP,
    "MolWt": Descriptors.MolWt,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "AromaticProportion": aromatic_proportion}
 
def featurise(molecules):
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([fp_gen.GetCountFingerprintAsNumPy(molecule) for molecule in molecules])
 
def featurise_descriptors_only(molecules, descriptor_functions):
    return np.array([[f(molecule) for f in descriptor_functions.values()] for molecule in molecules])
