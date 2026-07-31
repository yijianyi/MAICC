# MAICC: Multi-agent In-context Coordination via Decentralized Memory Retrieval

This is the implementation of the AAAI 2026 paper "Multi-agent In-context Coordination via Decentralized Memory Retrieval". The appendix of the paper is available at this [MAICC_supp.pdf](https://www.lamda.nju.edu.cn/jiangt/MAICC_supp.pdf).

### Install Python environment

Install Python environment with conda:

```bash
conda create -n MAICC python=3.8 -y
conda activate MAICC 
pip install pip==24.0
pip install setuptools==63.2.0
pip install wheel==0.38.4
pip install -r requirements.txt
pip install -e lb-foraging/
pip install -e smacv2/
```

### Install StarCraft

```bash
bash install_sc2.sh
```

### Run experiments

We provide the data link to the full dataset by this [Google Drive URL](https://drive.google.com/file/d/1V1YoCFE2bfnmA_whSyYgbuyXz_8zW5BY/view?usp=drive_link). Unzip datasets.zip into the MAICC/datasets folder.

You can run the following scripts to perform experiments in the paper.
```bash
bash lbf_small_train.sh
bash lbf_train.sh
bash sc2v2_train.sh
```

### Train on TABX offline data

TABX uses a separate streaming adapter so the original Gymma and SMACv2
training paths remain unchanged. Server preparation, stage-0, retrieval index,
and stage-1 commands are documented in
[`docs/TABX_TRAINING.md`](docs/TABX_TRAINING.md).
