# LUSCgnosis-GNN

LUSCgnosis-GNN is the official code implementation of a research project developing a prognostic prediction model for lung squamous cell carcinoma (LUSC). The model is built upon a **graph neural network (Graph Attention Network, GAT)**, learns patient-level risk representations on a graph constructed from a gene co-expression network, and outputs risk scores (hazard scores) for survival analysis.

---

## Code files

This project contains 4 core code files:

- `build_wgcna_adjacency.R`: Construct a gene co-expression network from gene expression data (WGCNA-style) and export the adjacency matrix.
- `train_survival_gat.py`: Load expression data and the adjacency matrix, train a GAT-based survival prediction model, and save the model weights and training configuration.
- `infer_survival_risk.py`: Load a pretrained model, run inference on a given patient cohort, and output a risk score (hazard score) for each patient.
- `explain_survival_shapley.py`: Use a Monte Carlo permutation-sampling Shapley approximation to estimate node-level contributions on a single-patient graph, explaining key nodes that influence risk predictions.

---

## Data format

Model training and inference use a patient-by-gene expression matrix (non-log-transformed TPM or normalized microarray expression), where **each row corresponds to one patient**. The feature dimensionality used for graph modeling is **320** (excluding `sample_id` and survival label columns):

- **lncRNA features**: columns 1–80 of the expression part (80d)
- **mRNA features**: columns 81–320 of the expression part (240d)

**Example CSV** (structure and column order only):

```csv
sample_id,lnc_001,lnc_002,...,lnc_080,mrna_001,mrna_002,...,mrna_240,OS,OS_time
P001,12.34,0.00,...,1.23,5.67,8.90,...,0.12,1,365
P002,0.56,1.78,...,0.00,3.21,4.56,...,7.89,0,420
```

Where:

- `sample_id`: sample/patient ID
- `OS`: event indicator (1=event occurred, 0=censored)
- `OS_time`: survival time

---

## Workflow

1. **Build adjacency matrix**  

```bash
Rscript build_wgcna_adjacency.R
```

   Build an adjacency matrix based on the training-set expression matrix and output `adjacency_matrix.csv`.

2. **Train the model**  

```bash
python train_survival_gat.py
```
   
   Train the GAT survival model and save a checkpoint (including model parameters and training configuration).

3. **Inference**  

```bash
python infer_survival_risk.py
```

   Load the checkpoint and output risk scores (hazard scores) for the specified dataset.

4. **Model explanation**  
```base
python explain_survival_shapley.py
```

   Run node-level Shapley explanations for specified patients and estimate each node's contribution to the predicted hazard.

> Before running, replace the `REPLACE_WITH_*` path placeholders in the scripts with your local paths.

---

## Outputs

- **Patient risk score (hazard score / `hazard_score`)**: a continuous risk value predicted by the model; higher values indicate higher predicted risk.
- `infer_survival_risk.py` saves a prediction CSV under `results_dir` (including `sample_id` and risk fields such as `hazard_pred`/`hazard_score`).
- `explain_survival_shapley.py` saves per-patient node contribution results (CSV) under `results_dir`.

---

## Dependencies

### Python

- Python 3.11
- `numpy==1.26.4`
- `torch==2.2.2`
- `pandas==2.2.3`

### R

- R 4.5.0
- `WGCNA==1.73`

---

## Acknowledgements and References

The implementation of this project draws upon the following open-source projects and their associated research papers, to which we extend our gratitude:

- **GitHub**: [`TencentAILabHealthcare/MLA-GNN`](https://github.com/TencentAILabHealthcare/MLA-GNN)
- **Reference**: Xing, X., Yang, F., Li, H., Zhang, J., Zhao, Y., Gao, M., Huang, J., & Yao, J. (2022). *Multi-level attention graph neural network based on co-expression gene modules for disease diagnosis and prognosis*. Bioinformatics.

---

## Disclaimer

This code is for research purposes only and does not constitute medical advice. The authors assume no responsibility for any direct or indirect consequences arising from the use of this project. Users are responsible for ensuring that their data processing, study design, and result interpretation comply with the ethical and regulatory requirements of their institutions and target journals.




