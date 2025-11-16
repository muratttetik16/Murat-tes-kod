#!/usr/bin/env python3
"""
Script to generate YOLO-style loss graphs from results.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_yolo_loss(csv_file='results.csv', output_file='loss_graph.png'):
    """
    Read results.csv and plot training and validation losses similar to YOLO output
    
    Args:
        csv_file: Path to the results.csv file
        output_file: Path to save the output graph
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, sep='\t')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Extract epoch numbers (filter out rows with NaN for all loss values)
    valid_rows = df.dropna(subset=['train/box_loss', 'train/cls_loss', 'train/dfl_loss'], how='all')
    epochs = valid_rows['epoch'].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('YOLO Training Loss Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: Train Box Loss
    ax = axes[0, 0]
    train_box_loss = valid_rows['train/box_loss'].values
    ax.plot(epochs, train_box_loss, 'b-', linewidth=2, label='train/box_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train Box Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Train Class Loss
    ax = axes[0, 1]
    train_cls_loss = valid_rows['train/cls_loss'].values
    ax.plot(epochs, train_cls_loss, 'g-', linewidth=2, label='train/cls_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train Classification Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Train DFL Loss
    ax = axes[0, 2]
    train_dfl_loss = valid_rows['train/dfl_loss'].values
    ax.plot(epochs, train_dfl_loss, 'r-', linewidth=2, label='train/dfl_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train DFL Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Validation Box Loss
    ax = axes[1, 0]
    val_data = df.dropna(subset=['val/box_loss'])
    val_epochs = val_data['epoch'].values
    val_box_loss = val_data['val/box_loss'].values
    ax.plot(val_epochs, val_box_loss, 'b--', linewidth=2, label='val/box_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Box Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Validation Class Loss
    ax = axes[1, 1]
    val_cls_loss = val_data['val/cls_loss'].values
    ax.plot(val_epochs, val_cls_loss, 'g--', linewidth=2, label='val/cls_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Classification Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Validation DFL Loss
    ax = axes[1, 2]
    val_dfl_loss = val_data['val/dfl_loss'].values
    ax.plot(val_epochs, val_dfl_loss, 'r--', linewidth=2, label='val/dfl_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation DFL Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Loss graph saved to: {output_file}")
    
    # Also create a combined loss plot
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Plot all training losses
    ax2.plot(epochs, train_box_loss, 'b-', linewidth=2, label='train/box_loss', alpha=0.7)
    ax2.plot(epochs, train_cls_loss, 'g-', linewidth=2, label='train/cls_loss', alpha=0.7)
    ax2.plot(epochs, train_dfl_loss, 'r-', linewidth=2, label='train/dfl_loss', alpha=0.7)
    
    # Plot validation losses
    ax2.plot(val_epochs, val_box_loss, 'b--', linewidth=2, label='val/box_loss', alpha=0.7)
    ax2.plot(val_epochs, val_cls_loss, 'g--', linewidth=2, label='val/cls_loss', alpha=0.7)
    ax2.plot(val_epochs, val_dfl_loss, 'r--', linewidth=2, label='val/dfl_loss', alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('YOLO Training and Validation Losses', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    combined_output = output_file.replace('.png', '_combined.png')
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"Combined loss graph saved to: {combined_output}")
    
    # Show statistics
    print("\n=== Loss Statistics ===")
    print(f"Final Train Box Loss: {train_box_loss[-1]:.5f}")
    print(f"Final Train Class Loss: {train_cls_loss[-1]:.5f}")
    print(f"Final Train DFL Loss: {train_dfl_loss[-1]:.5f}")
    if len(val_box_loss) > 0:
        print(f"Final Val Box Loss: {val_box_loss[-1]:.5f}")
        print(f"Final Val Class Loss: {val_cls_loss[-1]:.5f}")
        print(f"Final Val DFL Loss: {val_dfl_loss[-1]:.5f}")

if __name__ == '__main__':
    plot_yolo_loss()
