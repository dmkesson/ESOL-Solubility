import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from time import monotonic

df = pd.read_csv("data/delaney-processed.csv")

smiles_array = df["smiles"].str.strip().to_numpy()
molecules = [Chem.MolFromSmiles(s) for s in smiles_array]

y = df["measured log solubility in mols per litre"]

esol_predicted = df["ESOL predicted log solubility in mols per litre"]

def aromatic_proportion(molecule):
    num_aromatic = sum(1 for atom in molecule.GetAtoms() if atom.GetIsAromatic())
    
    return num_aromatic / molecule.GetNumHeavyAtoms()

descriptor_functions = {                                                #delaney equation descriptors
    "MolLogP": Descriptors.MolLogP,
    "MolWt": Descriptors.MolWt,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "AromaticProportion": aromatic_proportion
}

def featurise(molecules):
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([fp_gen.GetCountFingerprintAsNumPy(molecule) for molecule in molecules])

def featurise_descriptors_only(molecules, descriptor_functions):
    return np.array([[f(molecule) for f in descriptor_functions.values()] for molecule in molecules])

X_fp = featurise(molecules)
X_desc = featurise_descriptors_only(molecules, descriptor_functions)

X_fp_train, X_fp_test, X_desc_train, X_desc_test, y_train, y_test, esol_pred_train, esol_pred_test = train_test_split(X_fp, X_desc, y, esol_predicted, test_size=0.2, random_state=42)
X_both_train = np.hstack((X_fp_train, X_desc_train))
X_both_test = np.hstack((X_fp_test, X_desc_test))

fp_init_time = monotonic()
fp_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
fp_model.fit(X_fp_train, y_train)
fp_train_time = monotonic() - fp_init_time
fp_model_pred = fp_model.predict(X_fp_test)

desc_init_time = monotonic()
desc_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
desc_model.fit(X_desc_train, y_train)
desc_train_time = monotonic() - desc_init_time
desc_model_pred = desc_model.predict(X_desc_test)

both_init_time = monotonic()
both_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
both_model.fit(X_both_train, y_train)
both_train_time = monotonic() - both_init_time
both_model_pred = both_model.predict(X_both_test)

baseline_rmse = root_mean_squared_error(y_test, esol_pred_test)
fp_rmse       = root_mean_squared_error(y_test, fp_model_pred)
desc_rmse     = root_mean_squared_error(y_test, desc_model_pred)
both_rmse     = root_mean_squared_error(y_test, both_model_pred)
