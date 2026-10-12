import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from time import perf_counter
from collections import defaultdict

df = pd.read_csv("data/delaney-processed.csv")

smiles_array = df["smiles"].str.strip().to_numpy()
molecules = [Chem.MolFromSmiles(s) for s in smiles_array]
scaffolds = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in molecules])

y = df["measured log solubility in mols per litre"].to_numpy()

esol_predicted = df["ESOL predicted log solubility in mols per litre"].to_numpy()

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

X_fp = featurise(molecules)
X_desc = featurise_descriptors_only(molecules, descriptor_functions)

def scaffold_split(scaffolds, test_size=0.2):
    groups = defaultdict(list)
    for i, scaffold in enumerate(scaffolds):
        groups[scaffold].append(i)

    sorted_groups = sorted(groups.values(), key=len, reverse=True)  # sort by number of molecules with any particular scaffold
    train_rows = []
    test_rows = []
    train_size = len(scaffolds) * (1 - test_size)

    for group in sorted_groups:
        if len(train_rows) + len(test_rows) <= train_size:
            train_rows.extend(group)
        else:
            test_rows.extend(group)

    return np.array(train_rows), np.array(test_rows)

splits = {
    "random"  : train_test_split(np.arange(len(y)), test_size=0.2, random_state=42),
    "scaffold": scaffold_split(scaffolds)}

def make_experiments(X_fp_train, X_fp_test, X_desc_train, X_desc_test):
    
    X_both_train = np.hstack((X_fp_train, X_desc_train))
    X_both_test = np.hstack((X_fp_test, X_desc_test))
    
    experiments = {
        "Linear"        :     (LinearRegression(), X_desc_train, X_desc_test),
        "RF descriptors":     (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), X_desc_train, X_desc_test),
        "RF fingerprint":     (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), X_fp_train, X_fp_test),
        "RF both"       :     (RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1), X_both_train, X_both_test)}
    return experiments

def run_experiments(experiments, y_train, y_test, esol_pred_test):
    results = []
    trained_models = {}

    for name, (model, X_train, X_test) in experiments.items():
        start_time = perf_counter()
        model.fit(X_train, y_train)
        train_time = perf_counter() - start_time

        predictions = model.predict(X_test)
        results.append( {
            "model": name,
            "rmse" : root_mean_squared_error(y_test, predictions),
            "r2"   : r2_score(y_test, predictions),
            "time" : train_time})
        trained_models[name] = model

    results.insert(0, {
        "model": "ESOL equation (baseline)",
        "rmse" : root_mean_squared_error(y_test, esol_pred_test),
        "r2"   : r2_score(y_test, esol_pred_test)})

    return results, trained_models

all_results = []

for split_name, (train_rows, test_rows) in splits.items():
    experiments = make_experiments(X_fp[train_rows], X_fp[test_rows], X_desc[train_rows], X_desc[test_rows])
    results, models = run_experiments(experiments, y[train_rows], y[test_rows], esol_predicted[test_rows])
    for row in results:
        row["split"] = split_name
    all_results.extend(results)

print(pd.DataFrame(all_results))

for name, (tr, te) in splits.items():
    print(name, (scaffolds[te] == "").mean(), y[te].mean(), y[te].std())
