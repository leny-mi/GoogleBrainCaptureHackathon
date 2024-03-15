from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_latent_pca(latent_pca, y_data, annotations, pca, first_index=0, second_index=1, title="Latent Space after PCA"):
    fig, ax = plt.subplots(1, 2, figsize=(16, 3))

    # Generate a colormap that has as many colors as you have unique labels
    unique_labels = torch.unique(y_data.cpu())
    n_unique_labels = len(unique_labels)
    cmap = ListedColormap(plt.colormaps.get('tab10').colors[:n_unique_labels])

    # Map each label to a color
    label_to_color = {label.item(): cmap(i) for i, label in enumerate(unique_labels)}

    # Color each point in the scatter plot according to its label
    colors = [label_to_color[label.item()] for label in y_data.cpu()]

    # Plot the scatter plot
    ax[0].scatter(latent_pca[:, first_index], latent_pca[:, second_index], c=colors)
    ax[0].set_xlabel(f"Principal Component {first_index}")
    ax[0].set_ylabel(f"Principal Component {second_index}")
    ax[0].set_title(title)

    # Create legend
    legend_labels = [list(annotations.keys())[value] for value in unique_labels]
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=label_to_color[value.item()], markersize=10) for label, value in zip(legend_labels, unique_labels)]
    ax[0].legend(handles=legend_elements, loc='best', fontsize=6)

    # plot the explained variance ratio
    ax[1].plot(pca.explained_variance_ratio_)
    ax[1].set_xlabel("Principal Component")
    ax[1].set_ylabel("Explained Variance Ratio")
    ax[1].set_title("Explained Variance Ratio of Principal Components")

    plt.show()

def visualize_plot_from_eeg_data(df, start_time, window_size):
    # np.random.seed(0)
    # time = np.linspace(0, 100, 1000)  # 1000 time points from 0 to 100 seconds
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    # data = np.random.randn(1000, len(channels))  # Random data for 19 channels
    # df = pd.DataFrame(data, columns=channels)
    # df['time'] = time

    # Create the main figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(bottom=0.25)

    # Plot configuration
    ax.set_xlabel('Time (s)')
    ax.set_xlim(start_time, start_time + window_size)  # Initial x-axis limit

    # Calculate the vertical displacement and set the vertical limits
    n_rows = len(channels)
    dy = (df[channels].min().min() - df[channels].max().max()) * 0.7
    ax.set_ylim(-dy, n_rows * dy)

    # Plot each channel with an offset
    for i, channel in enumerate(channels):
        ax.plot(df['time'], df[channel] + i * dy, label=channel)

    # Set y-ticks to correspond to each channel, adjusting for the offsets
    ax.set_yticks(np.arange(0, n_rows * dy, dy))
    ax.set_yticklabels(channels)

    # plt.legend()
    # plt.show()
    return fig
