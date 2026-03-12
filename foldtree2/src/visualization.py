import torch
from colour import Color

def create_reconstruction_figure(data_sample, predictions, z_discrete, num_embeddings, 
                                epoch=None, save_path=None, figsize=(15, 10)):
    """
    Create reconstruction visualization from pre-computed predictions.
    Args:
        data_sample: Original data sample
        predictions: Dictionary of predictions from decoder
        z_discrete: Discrete embeddings
        num_embeddings: Size of embedding alphabet
        epoch: Epoch number (optional)
        save_path: Path to save figure
        figsize: Figure size
    Returns:
        (fig, metrics_dict) tuple
    """
    import numpy as np
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
    # Row 1: Predictions
    # AA predictions
    if 'aa' in predictions and predictions['aa'] is not None:
        aa_probs = torch.softmax(predictions['aa'], dim=-1).cpu().numpy()
        im0 = axs[0, 0].imshow(aa_probs.T, cmap='hot', aspect='auto')
        axs[0, 0].set_title(f"{epoch_str}AA Predictions")
        axs[0, 0].set_xlabel('Residue Index')
        axs[0, 0].set_ylabel('AA Type')
        fig.colorbar(im0, ax=axs[0, 0])
    # Contact predictions
    if 'edge_probs' in predictions and predictions['edge_probs'] is not None:
        edge_probs = predictions['edge_probs'].cpu().numpy()
        im1 = axs[0, 1].imshow(1 - edge_probs, cmap='hot', interpolation='nearest')
        axs[0, 1].set_title(f"{epoch_str}Predicted Contacts")
        fig.colorbar(im1, ax=axs[0, 1])
    # Embedding sequence
    if z_discrete is not None:
        ord_colors = Color("red").range_to(Color("blue"), num_embeddings)
        ord_colors = np.array([c.get_rgb() for c in ord_colors])
        sequence_colors = ord_colors[z_discrete.cpu().numpy()]
        max_width = 64
        seq_len = len(sequence_colors)
        rows = int(np.ceil(seq_len / max_width))
        canvas = np.ones((rows, max_width, 3))
        for i in range(rows):
            start = i * max_width
            end = min((i + 1) * max_width, seq_len)
            row_colors = sequence_colors[start:end]
            canvas[i, :len(row_colors), :] = row_colors
        axs[0, 2].imshow(canvas, aspect='auto')
        axs[0, 2].set_title('Embedding Sequence')
        axs[0, 2].axis('off')
    # Row 2: Additional predictions
    # Angles
    if 'angles' in predictions and predictions['angles'] is not None:
        angles = predictions['angles'].cpu().numpy()
        for i in range(min(3, angles.shape[1])):
            axs[1, 0].plot(angles[:, i], label=f'Angle {i}', alpha=0.7)
        axs[1, 0].set_title('Predicted Angles')
        axs[1, 0].legend()
        axs[1, 0].set_xlabel('Residue Index')
        axs[1, 0].set_ylabel('Angle (radians)')
    # Secondary structure
    if 'ss_pred' in predictions and predictions['ss_pred'] is not None:
        ss_pred = torch.argmax(predictions['ss_pred'], dim=-1).cpu().numpy()
        ss_colors = Color("red").range_to(Color("blue"), 3)
        ss_colors = np.array([c.get_rgb() for c in ss_colors])
        ss_sequence = ss_colors[ss_pred]
        max_width = 64
        rows = int(np.ceil(len(ss_sequence) / max_width))
        canvas = np.ones((rows, max_width, 3))
        for i in range(rows):
            start = i * max_width
            end = min((i + 1) * max_width, len(ss_sequence))
            row_colors = ss_sequence[start:end]
            canvas[i, :len(row_colors), :] = row_colors
        axs[1, 1].imshow(canvas, aspect='auto')
        axs[1, 1].set_title('Predicted SS')
        axs[1, 1].axis('off')
    # Edge logits heatmap
    if 'edge_logits' in predictions and predictions['edge_logits'] is not None:
        edge_logits = predictions['edge_logits']
        if edge_logits.dim() == 3:
            # Sum over categories if multi-dimensional
            edge_logits = edge_logits.sum(dim=-1)
        im2 = axs[1, 2].imshow(edge_logits.cpu().numpy(), cmap='hot', interpolation='nearest')
        axs[1, 2].set_title('Edge Logits')
        fig.colorbar(im2, ax=axs[1, 2])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    # Compute basic metrics
    metrics = {
        'num_residues': len(z_discrete) if z_discrete is not None else 0,
        'has_aa': 'aa' in predictions,
        'has_contacts': 'edge_probs' in predictions,
        'has_angles': 'angles' in predictions,
        'has_ss': 'ss_pred' in predictions
    }
    return fig, metrics

def visualize_batch_reconstructions(encoder, decoder, data_samples, device, num_embeddings, 
                                    converter, epoch=None, save_dir=None, max_samples=4):
    """
    Visualize reconstructions for a batch of samples.
    Args:
        encoder: Trained encoder
        decoder: Trained decoder
        data_samples: List of data samples
        device: PyTorch device
        num_embeddings: Number of discrete embeddings
        converter: PDB2PyG converter
        epoch: Current epoch (for titles)
        save_dir: Directory to save figures
        max_samples: Maximum number of samples to visualize
    Returns:
        List of (figure, metrics) tuples
    """
    import os
    encoder.eval()
    decoder.eval()
    # Limit number of samples
    data_samples = data_samples[:max_samples]
    # Encode all samples
    z_batch = []
    with torch.no_grad():
        for data in data_samples:
            data = data.to(device)
            z, _ = encoder(data)
            z_discrete = encoder.vector_quantizer.discretize_z(z.detach())[0]
            z_batch.append(z_discrete)
    # Batch decode
    results = decoder.decode_batch_with_contacts(z_batch, device, converter, encoder)
    # Visualize each sample
    figures_and_metrics = []
    for idx, (data_sample, result, z_discrete) in enumerate(zip(data_samples, results, z_batch)):
        try:
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'reconstruction_sample{idx}.png')
            fig, metrics = create_reconstruction_figure(
                data_sample, result, z_discrete, num_embeddings, 
                epoch=epoch, save_path=save_path
            )
            figures_and_metrics.append((fig, metrics))
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            continue
    encoder.train()
    decoder.train()
    return figures_and_metrics
"""
visualization.py

Reusable visualization functions for FoldTree2 experiments and training scripts.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(train_losses, val_losses=None, title="Loss Curve", ylabel="Loss", xlabel="Epoch", save_path=None, show=True):
    """
    Plot training (and optionally validation) loss curves.
    Args:
        train_losses (list or np.ndarray): Training loss values.
        val_losses (list or np.ndarray, optional): Validation loss values.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        xlabel (str): X-axis label.
        save_path (str, optional): If provided, saves the figure to this path.
        show (bool): Whether to display the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_metric_curve(metric_values, title="Metric Curve", ylabel="Metric", xlabel="Epoch", save_path=None, show=True):
    """
    Plot a generic metric curve (e.g., accuracy, F1, etc.).
    Args:
        metric_values (list or np.ndarray): Metric values per epoch.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        xlabel (str): X-axis label.
        save_path (str, optional): If provided, saves the figure to this path.
        show (bool): Whether to display the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(metric_values, label=ylabel, color="green")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_multiple_curves(curves, labels, title="Curves", ylabel="Value", xlabel="Epoch", save_path=None, show=True):
    """
    Plot multiple curves on the same figure.
    Args:
        curves (list of list or np.ndarray): List of value sequences.
        labels (list of str): Labels for each curve.
        title (str): Plot title.
        ylabel (str): Y-axis label.
        xlabel (str): X-axis label.
        save_path (str, optional): If provided, saves the figure to this path.
        show (bool): Whether to display the plot.
    """
    plt.figure(figsize=(8, 5))
    for curve, label in zip(curves, labels):
        plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
