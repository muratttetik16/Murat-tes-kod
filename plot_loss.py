#!/usr/bin/env python3
"""
Script to generate YOLOv8-style loss graphs from results.csv
Creates visualization matching the official Ultralytics YOLOv8 results.png format
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_yolo_loss(csv_file='results.csv', output_file='results.png'):
    """
    Read results.csv and plot training and validation losses in YOLOv8 style
    
    Args:
        csv_file: Path to the results.csv file
        output_file: Path to save the output graph (default: results.png)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, sep='\t')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Extract data for training (all rows with valid training data)
    train_data = df.dropna(subset=['train/box_loss', 'train/cls_loss', 'train/dfl_loss'], how='all')
    train_epochs = train_data['epoch'].values
    
    # Extract data for validation (only rows with valid validation data)
    val_data = df.dropna(subset=['val/box_loss', 'val/cls_loss', 'val/dfl_loss'], how='all')
    val_epochs = val_data['epoch'].values
    
    # Create figure with 2x5 grid layout (similar to YOLOv8 results.png)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('YOLOv8x Model Eğitim Kayıp Grafikleri', fontsize=18, fontweight='bold', y=0.98)
    
    # Color scheme for consistency
    colors = {'box': '#1f77b4', 'cls': '#ff7f0e', 'dfl': '#2ca02c'}
    
    # Row 1: Training losses and metrics
    # Plot 1: train/box_loss
    ax = axes[0, 0]
    ax.plot(train_epochs, train_data['train/box_loss'].values, color=colors['box'], linewidth=2)
    ax.set_title('train/box_loss', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 2: train/cls_loss
    ax = axes[0, 1]
    ax.plot(train_epochs, train_data['train/cls_loss'].values, color=colors['cls'], linewidth=2)
    ax.set_title('train/cls_loss', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 3: train/dfl_loss
    ax = axes[0, 2]
    ax.plot(train_epochs, train_data['train/dfl_loss'].values, color=colors['dfl'], linewidth=2)
    ax.set_title('train/dfl_loss', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 4: metrics/precision(B)
    ax = axes[0, 3]
    precision_data = train_data.dropna(subset=['metrics/precision(B)'])
    if len(precision_data) > 0:
        ax.plot(precision_data['epoch'].values, precision_data['metrics/precision(B)'].values, 
                color='#d62728', linewidth=2)
    ax.set_title('metrics/precision(B)', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 5: metrics/recall(B)
    ax = axes[0, 4]
    recall_data = train_data.dropna(subset=['metrics/recall(B)'])
    if len(recall_data) > 0:
        ax.plot(recall_data['epoch'].values, recall_data['metrics/recall(B)'].values, 
                color='#9467bd', linewidth=2)
    ax.set_title('metrics/recall(B)', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Row 2: Validation losses and metrics
    # Plot 6: val/box_loss
    ax = axes[1, 0]
    if len(val_data) > 0:
        ax.plot(val_epochs, val_data['val/box_loss'].values, color=colors['box'], linewidth=2)
    ax.set_title('val/box_loss', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 7: val/cls_loss
    ax = axes[1, 1]
    if len(val_data) > 0:
        ax.plot(val_epochs, val_data['val/cls_loss'].values, color=colors['cls'], linewidth=2)
    ax.set_title('val/cls_loss', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 8: val/dfl_loss
    ax = axes[1, 2]
    if len(val_data) > 0:
        ax.plot(val_epochs, val_data['val/dfl_loss'].values, color=colors['dfl'], linewidth=2)
    ax.set_title('val/dfl_loss', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 9: metrics/mAP50(B)
    ax = axes[1, 3]
    map50_data = df.dropna(subset=['metrics/mAP50(B)'])
    if len(map50_data) > 0:
        ax.plot(map50_data['epoch'].values, map50_data['metrics/mAP50(B)'].values, 
                color='#8c564b', linewidth=2)
    ax.set_title('metrics/mAP50(B)', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Plot 10: metrics/mAP50-95(B)
    ax = axes[1, 4]
    map50_95_data = df.dropna(subset=['metrics/mAP50-95(B)'])
    if len(map50_95_data) > 0:
        ax.plot(map50_95_data['epoch'].values, map50_95_data['metrics/mAP50-95(B)'].values, 
                color='#e377c2', linewidth=2)
    ax.set_title('metrics/mAP50-95(B)', fontsize=11, fontweight='bold')
    ax.set_xlabel('epoch', fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=8)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"YOLOv8 style results graph saved to: {output_file}")
    
    # Show statistics
    print("\n=== Training Statistics ===")
    if len(train_data) > 0:
        print(f"Epochs trained: {len(train_data)}")
        print(f"Final train/box_loss: {train_data['train/box_loss'].iloc[-1]:.5f}")
        print(f"Final train/cls_loss: {train_data['train/cls_loss'].iloc[-1]:.5f}")
        print(f"Final train/dfl_loss: {train_data['train/dfl_loss'].iloc[-1]:.5f}")
    
    if len(val_data) > 0:
        print(f"\n=== Validation Statistics ===")
        print(f"Final val/box_loss: {val_data['val/box_loss'].iloc[-1]:.5f}")
        print(f"Final val/cls_loss: {val_data['val/cls_loss'].iloc[-1]:.5f}")
        print(f"Final val/dfl_loss: {val_data['val/dfl_loss'].iloc[-1]:.5f}")
    
    if len(map50_95_data) > 0:
        print(f"\n=== Best Metrics ===")
        best_map50_95 = map50_95_data['metrics/mAP50-95(B)'].max()
        print(f"Best mAP50-95: {best_map50_95:.5f}")

if __name__ == '__main__':
    plot_yolo_loss()
