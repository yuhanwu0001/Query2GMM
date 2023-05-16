# Query2GMM
This is the code of paper **Query2GMM: Learning Representation with Gaussian Mixture Model for Reasoning over Knowledge Graphs**.

## Requirments
* Python 3.8
* PyTorch 1.11
* tqdm

## Data
* data for EPFO queries provided by: <https://github.com/hyren/query2box>
* data for queries with negation provided by: <https://github.com/snap-stanford/KGReasoning>

## Run
* ./Q2GMM/codes/example.sh

## Baseline codes
* BetaE: <https://github.com/snap-stanford/KGReasoning>
* PERM: <https://github.com/Akirato/PERM-GaussianKG>
* NMP-QEM: We re-implement the NMP-QEM for comparison since we don't obtain the source code. <https://anonymous.4open.science/r/Re-Implement_of_NMP-QEM-5E85>
* Query2Particles: <https://github.com/HKUST-KnowComp/query2particles>
* LMPNN: <https://github.com/HKUST-KnowComp/LMPNN>
