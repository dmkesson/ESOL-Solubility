# Predicting aqueous solubility (ESOL)

Compares the published ESOL equation (Delaney, 2004) with a refitted linear model and random forests on Morgan count fingerprints, Delaney's four descriptors, and both, using random and scaffold splits repeated over 10 seeds. See `esol_multiseed.ipynb`.

**Key result:** a random forest model trained on both fingerprints and descriptors has less error than the ESOL equation in Delaney's paper(RMSE 0.65 vs 0.89 log mol/L on random splits). When using scaffold splits, the advantage decreases (RMSE 0.91 vs 0.99), and fingerprint-only models generalise the worst (RMSE 1.35).

**Limitations:** duplicate structures were not merged, so some data leakage could occur in a random split. Acyclic and benzene-scaffold molecules (about half the dataset) were too numerous to fit in a scaffold test set. This means that the scaffold results only reflect performance on less common ring systems (though that's the point of a scaffold split).

**Data:** `delaney-processed.csv` from [DeepChem](https://github.com/deepchem/deepchem) (MIT licence, © 2017 PandeLab), originally from Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly from Molecular Structure. J. Chem. Inf. Comput. Sci., 44(3), 1000–1005. https://doi.org/10.1021/ci034243x
