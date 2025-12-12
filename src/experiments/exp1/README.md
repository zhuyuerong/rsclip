# SurgeryCLIP + AAF + p2p Propagation Experiment

This experiment implements an enhanced version of SurgeryCLIP with Adaptive Attention Fusion (AAF) and patch-to-patch (p2p) propagation for improved Class Activation Map (CAM) generation.

## Overview

The experiment enhances SurgeryCLIP by:
1. **AAF (Adaptive Attention Fusion)**: Fuses attention from the last 6 layers of SurgeryCLIP's dual-path (VV and original)
2. **Patch-to-patch propagation**: Uses fused attention for spatial propagation to improve CAM quality
3. **Lightweight training**: Only trains AAF parameters (~13 parameters) while freezing CLIP

## Architecture

```
SurgeryCLIP (frozen)
    ↓
Last 6 layers → VV attention + Original attention
    ↓
AAF Layer (trainable)
    ↓
Fused attention [B, N², N²]
    ↓
CAM Generator
    ↓
Initial CAM (patch-text similarity)
    ↓
p2p Propagation
    ↓
Final CAM [B, C, N, N]
```

## Project Structure

```
exp1/
├── models/
│   ├── __init__.py
│   ├── aaf.py                      # AAF layer implementation
│   ├── cam_generator.py            # CAM generator with p2p propagation
│   └── surgery_aaf.py              # Main model
├── utils/
│   ├── __init__.py
│   ├── data.py                     # Data loading (DIOR dataset)
│   ├── visualization.py            # CAM visualization
│   └── metrics.py                  # Evaluation metrics
├── configs/
│   └── config.yaml                 # Configuration file
├── train.py                        # Training script
├── eval.py                         # Evaluation script
├── checkpoints/                    # Saved model weights
└── outputs/                        # Evaluation results and visualizations
```

## Setup

1. **Install dependencies** (if not already installed):
   ```bash
   pip install torch torchvision pillow matplotlib numpy pyyaml tqdm
   ```

2. **Prepare DIOR dataset**:
   - Place DIOR dataset in `datasets/DIOR/` or specify path in config
   - Dataset structure should be:
     ```
     DIOR/
     ├── images/
     │   ├── trainval/
     │   └── test/
     ├── annotations/
     │   └── horizontal/
     └── splits/
     ```

3. **Download CLIP weights**:
   - Place CLIP checkpoint in `checkpoints/` directory
   - Update `clip_weights_path` in `configs/config.yaml`

## Usage

### Training

```bash
cd src/experiments/exp1
python train.py
```

The training script will:
- Load pre-trained SurgeryCLIP
- Freeze CLIP parameters
- Train only AAF parameters
- Save checkpoints to `exp1/checkpoints/`

### Evaluation

```bash
python eval.py
```

The evaluation script will:
- Load trained AAF weights
- Evaluate on test set
- Generate CAM visualizations
- Save metrics to `exp1/outputs/`

## Configuration

Edit `configs/config.yaml` to customize:

- **Model**: CLIP model name, checkpoint path, number of layers
- **Training**: Epochs, batch size, learning rate, weight decay
- **Data**: Dataset name, root path, number of workers
- **Output**: Checkpoint and output directories

## Key Features

### AAF Layer
- Learns per-layer weights for VV and original paths
- Learns mixing coefficient between paths
- Only ~13 trainable parameters

### CAM Generation
1. **Initial CAM**: Patch-text similarity
2. **p2p Propagation**: Uses fused attention to propagate activations spatially

### Training Strategy
- Freezes entire CLIP model (~100M parameters)
- Trains only AAF (~13 parameters)
- Low memory footprint
- Fast training

## Expected Results

- **Initial CAM**: May have partial activations
- **After p2p propagation**: Activations spread to similar patches, complete target coverage
- **AAF effect**: Learns optimal layer weight combination

## Notes

- The experiment modifies `clip_surgery_model.py` to expose attention weights
- Make sure SurgeryCLIP is properly loaded before training
- CAM visualizations are saved to `outputs/visualizations/`
- Metrics are saved to `outputs/results.json`

## Troubleshooting

1. **Import errors**: Make sure paths are correct and SurgeryCLIP is in the expected location
2. **Dataset not found**: Update `dataset_root` in config or place dataset in default location
3. **CUDA out of memory**: Reduce batch size in config
4. **Attention not collected**: Ensure SurgeryCLIP model has been modified to store attention weights





