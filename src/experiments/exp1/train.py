#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for SurgeryCLIP + AAF + p2p experiment
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.surgery_aaf import create_surgery_aaf_model
from utils.data import get_dataloader


def train(config):
    """
    Train SurgeryAAF model
    
    Args:
        config: Configuration dictionary
    """
    device = config['device']
    
    # ===== 1. Load pre-trained SurgeryCLIP =====
    print("=" * 80)
    print("Loading SurgeryCLIP...")
    print("=" * 80)
    
    checkpoint_path = config['clip_weights_path']
    if not os.path.isabs(checkpoint_path):
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        checkpoint_path = project_root / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
    
    model, preprocess = create_surgery_aaf_model(
        checkpoint_path=checkpoint_path,
        device=device,
        num_layers=config.get('num_layers', 6)
    )
    
    # ===== 2. Freeze CLIP parameters, only train AAF =====
    print("\nFreezing CLIP parameters...")
    for param in model.clip.parameters():
        param.requires_grad = False
    
    # Only AAF parameters are trainable
    for param in model.aaf.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.aaf.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ===== 3. Optimizer =====
    optimizer = torch.optim.AdamW(
        model.aaf.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # ===== 4. Loss function =====
    criterion = nn.BCEWithLogitsLoss()
    
    # ===== 5. Data loading =====
    print("\nLoading data...")
    train_loader = get_dataloader(
        dataset_name=config['dataset'],
        root=config.get('dataset_root'),
        split='trainval',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    val_loader = get_dataloader(
        dataset_name=config['dataset'],
        root=config.get('dataset_root'),
        split='test',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # ===== 6. Training loop =====
    print("\n" + "=" * 80)
    print("Start training...")
    print("=" * 80)
    
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['num_epochs']}]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)  # [B, C]
            text_queries = batch['text_queries'][0]  # All classes (same for all samples)
            
            # Forward pass
            cam, aux = model(images, text_queries)
            # cam: [B, C, N, N]
            
            # Extract classification scores from CAM (global max pooling)
            cam_scores = cam.flatten(2).max(dim=2)[0]  # [B, C]
            
            # Compute loss
            loss = criterion(cam_scores, labels.float())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predictions = (torch.sigmoid(cam_scores) > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.numel()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.4f}'
            })
            
            # Logging
            if batch_idx % config['log_interval'] == 0:
                print(f"\nEpoch [{epoch+1}/{config['num_epochs']}] "
                      f"Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} "
                      f"Acc: {train_correct/train_total:.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['images'].to(device)
                labels = batch['labels'].to(device)
                text_queries = batch['text_queries'][0]
                
                cam, _ = model(images, text_queries)
                cam_scores = cam.flatten(2).max(dim=2)[0]
                
                loss = criterion(cam_scores, labels.float())
                val_loss += loss.item()
                
                predictions = (torch.sigmoid(cam_scores) > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.numel()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*80}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.aaf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
            }, checkpoint_dir / 'best_model.pth')
            print("✅ Best model saved!")
        
        # Periodic checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.aaf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"✅ Checkpoint saved: epoch_{epoch+1}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    # Load configuration
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    train(config)





