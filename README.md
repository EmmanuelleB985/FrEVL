# FrEVL: Frozen Pretrained Embeddings for Efficient Vision-Language Understanding

**[ICCVW 2025 SafeMM-AI]** Official implementation of "Leveraging Frozen Pretrained Embeddings for Efficient Vision-Language Understanding"

📄 **Paper**: [arXiv:2508.04469](https://arxiv.org/pdf/2508.04469)

## Overview

FrEVL is a lightweight framework that leverages frozen CLIP embeddings for efficient vision-language understanding. By keeping pretrained encoders frozen and training only a compact fusion network (68.4M parameters), FrEVL achieves 85-95% of state-of-the-art performance while providing significant efficiency gains.

## Installation

```bash

# Create environment
conda create -n frevl python=3.8
conda activate frevl

# Install dependencies
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install scikit-learn matplotlib pandas numpy
```

## Dataset Preparation

Download and prepare the datasets:

```bash
# Download all datasets
python prepare_datasets.py --dataset all --data-dir ./data

# Or download specific dataset
python prepare_datasets.py --dataset coco --data-dir ./data
python prepare_datasets.py --dataset vqa --data-dir ./data
python prepare_datasets.py --dataset snli-ve --data-dir ./data

# Optional: Pre-compute embeddings for faster training
python prepare_datasets.py --dataset all --cache-embeddings
```

**Note**: SNLI-VE requires manual download of Flickr30k images from [here](https://shannon.cs.illinois.edu/DenotationGraph/).

## Training

### Quick Start

```bash
# Train FrEVL on VQA v2
python train.py
```

## Evaluation

```bash
# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Project Structure

```
FrEVL/
├── train.py                 # Main training script
├── evaluate.py             # Evaluation script
├── prepare_datasets.py     # Dataset download and preparation
├── configs/               # Configuration files
├── checkpoints_regularized/  # Saved model checkpoints
├── logs/                   # Training logs
└── data/                    # Dataset directory
    ├── coco/
    ├── vqa/
    └── snli-ve/
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{bourigault2025frevl,
  title={FrEVL: Leveraging Frozen Pretrained Embeddings for Efficient Vision-Language Understanding},
  author={Bourigault, Emmanuelle and Bourigault, Pauline},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the authors of CLIP, COCO, VQA v2, and SNLI-VE for making their datasets and models publicly available.