Code for“Physics-informed deep leaning for generalizable structured illumination microscopy reconstruction”
# PID-SIM (Code Release)

This repository provides an implementation of **PID-SIM** with a two-stage reproduction pipeline:

1. **Stage I (Pretraining)** on **synthetic HeLa** data: `main_syn.py`  
2. **Stage II (Fine-tuning)** on **BioSR** (public dataset): `main_real.py` (initialized from Stage I weights)

All dataset paths and checkpoint paths are configured via **command-line arguments (args)**.

---

## Environment (Tested)

- OS: **Ubuntu 20.04.4 LTS** (focal)
- Kernel: **5.4.0-42-generic**
- Python: **3.8.20** (Conda env: `DT`)
- GPU: **8× NVIDIA GPU (24GB-class)**
- NVIDIA Driver: **470.74** (Driver-reported CUDA: **11.4**)
- CUDA Toolkit (`nvcc`): **NOT installed**
- PyTorch: **2.4.1+cu121**
  - `torch.version.cuda = 12.1`
  - ⚠️ **CUDA is NOT available in this tested environment** because the installed PyTorch CUDA runtime (cu121) is incompatible with the old NVIDIA driver (470.74 / CUDA 11.4 era).

### CUDA Compatibility Note

To run on GPU, your **NVIDIA driver** must be compatible with the **CUDA runtime** shipped with your PyTorch build.

- Option A (recommended): **upgrade NVIDIA driver** to support CUDA 12.x (for `torch 2.4.x + cu121`)
- Option B: keep the current driver and **install a PyTorch build compiled for CUDA 11.x** (cu11x) that matches your cluster/driver constraints

> If CUDA is unavailable, training may still run on CPU but will be extremely slow and not recommended.

---

## Installation

We recommend using **Miniconda/Anaconda** to manage the environment.

### Step 1. Create and activate conda env

conda create -n pid_sim python=3.8 -y
conda activate pid_sim

### Step 2. Install PyTorch (choose ONE)

A) GPU (recommended; match your driver/runtime)
Example (CUDA 11.8 build):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
B) CPU-only (debug only)
pip install torch torchvision torchaudio
Pretrained Weights
We provide the Stage I pretrained checkpoint under:

checkpoint/

During Stage II fine-tuning, pass the pretrained checkpoint path via args (see examples below).

Datasets
Stage I: Synthetic HeLa
Prepare the synthetic HeLa dataset and provide its path via args when running main_syn.py.

Stage II: BioSR (Public)
BioSR can be downloaded from its official source. Provide the BioSR dataset path via args when running main_real.py.

Reproduction Pipeline
Stage I — Synthetic Pretraining (HeLa): main_syn.py
python main_syn.py \
  --data_root /abs/path/to/HeLa_syn \
  --save_dir checkpoint/pretrain_hela
--data_root: path to synthetic HeLa dataset

--save_dir: directory to save pretrained checkpoints

The exact argument names may differ. Run python main_syn.py --help to confirm the available args and replace the placeholders accordingly.

Stage II — Fine-tuning on BioSR: main_real.py
python main_real.py \
  --data_root /abs/path/to/BioSR \
  --pretrained checkpoint/<your_pretrained_ckpt>.pth \
  --save_dir checkpoint/finetune_biosr
--data_root: path to BioSR dataset

--pretrained: Stage I pretrained checkpoint under checkpoint/

--save_dir: directory to save fine-tuned checkpoints

The exact argument names may differ. Run python main_real.py --help to confirm the available args and replace the placeholders accordingly.

Quick Debug Checklist
Verify PyTorch sees GPU:

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
Verify driver:

nvidia-smi
If torch.cuda.is_available() is False, please check the driver ↔ PyTorch CUDA runtime compatibility described above.

Citation
To be updated.
