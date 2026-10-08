import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

esol = np.genfromtxt("data/delaney-processed.csv", delimiter=',', dtype=None)


def featurise(smiles):
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
