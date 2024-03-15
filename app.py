# app.py
import math
import mne
import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.getcwd() + '/')
from src.data.utils.eeg import get_raw
from src.data.processing import load_data_dict, get_data
from src.data.conf.eeg_annotations import braincapture_annotations
from src.data.conf.eeg_channel_picks import hackathon
from src.data.conf.eeg_channel_order import standard_19_channel
from src.data.conf.eeg_annotations import braincapture_annotations, tuh_eeg_artefact_annotations, tuh_eeg_index_to_articfact_annotations
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile
import torch
from tqdm import tqdm
from copy import deepcopy
from model.model import BendrEncoder
from model.model import Flatten
from sklearn.cluster import KMeans
from src.visualisation.visualisation import plot_latent_pca, visualize_plot_from_eeg_data
import icecream as ic
import xgboost as xgb
from plotly.graph_objs import Scatter, Layout, YAxis, Annotation, Annotations, Font, Data, Figure

max_length = lambda raw : int(raw.n_times / raw.info['sfreq']) 
DURATION = 10
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
models_directory = 'src/classifiers/'
n_labels = 5


def classify(X_new):
    loaded_models = []

    for i in range(n_labels):
        model_filename = os.path.join(models_directory, f"xgb_model_label_{i}.json")
        bst = xgb.Booster()  # Instantiate model
        bst.load_model(model_filename)  # Load model
        loaded_models.append(bst)
        print(f"Model loaded from {model_filename}")

    dnew = xgb.DMatrix(X_new)

    new_predictions = np.zeros((X_new.shape[0], n_labels))

    for i, model in enumerate(loaded_models):
        new_predictions[:, i] = model.predict(dnew)
    new_predictions_binary = (new_predictions > 0.5).astype(int)
    return new_predictions_binary

def generate_latent_representations(data, encoder, batch_size=5, device='cpu'):
    """ Generate latent representations for the given data using the given encoder.
    Args:
        data (np.ndarray): The data to be encoded.
        encoder (nn.Module): The encoder to be used.
        batch_size (int): The batch size to be used.
    Returns:
        np.ndarray: The latent representations of the given data.
    """
    data = data.to(device)

    latent_size = (1536, 4) # do not change this 
    latent = np.empty((data.shape[0], *latent_size))


    for i in tqdm(range(0, data.shape[0], batch_size)):
        latent[i:i+batch_size] = encoder(data[i:i+batch_size]).cpu().detach().numpy()

    return latent.reshape((latent.shape[0], -1))

def load_model(device='cpu'):
    """Loading BendrEncoder model
    Args:
        device (str): The device to be used.
    Returns:
        BendrEncoder (nn.Module): The model
    """

    # Initialize the model
    encoder = BendrEncoder()

    # Load the pretrained model
    encoder.load_state_dict(deepcopy(torch.load("encoder.pt", map_location=device)))
    encoder = encoder.to(device)

    return encoder

def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

def get_file_paths(edf_file_buffers):
    """
    input: edf_file_buffers: list of files uploaded by user

    output: paths: paths to the files
    """
    paths = []
    # make tempoary directory to store the files
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)
    for edf_file_buffer in edf_file_buffers:
        folder_name = os.path.join(temp_dir, edf_file_buffer.name[:4])
        make_dir(folder_name)
        # make tempoary file
        path = os.path.join(folder_name , edf_file_buffer.name)
        # write bytesIO object to file
        with open(path, 'wb') as f:
            f.write(edf_file_buffer.getvalue())

        paths.append(path)

    return temp_dir + '/', paths

def plot_clusters(components, labels):
    """
    input: 
        components: 2D array of the principal components
        labels: labels of the clusters
    
    output: None"""

    # Plot clusters
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for cluster_label in unique_labels:
        ax.scatter(components[labels == cluster_label, 0], components[labels == cluster_label, 1], label=f'Cluster {cluster_label}')

    ax.set_title('Clusters using PCA')
    ax.set_xlabel('Principal Component 0')
    ax.set_ylabel('Principal Component 1')
    ax.legend()

    st.pyplot(fig)

def plot_raw(raw, st, range_from, range_to):

    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)
    start, stop = raw.time_as_index([range_from, range_to])

    n_channels = min(len(picks), 20)
    data, times = raw[picks[:n_channels], start:stop]
    ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]

    step = 1. / n_channels
    kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

    # create objects for layout and traces
    layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
    traces = [Scatter(x=times, y=data.T[:, 0])]

    # loop over the channels
    for ii in range(1, n_channels):
        kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
        layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
        traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))

    # add channel names using Annotations
    annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper',   yref='y%d' % (ii + 1),
                                        text=ch_name, font=Font(size=9), showarrow=False)
                            for ii, ch_name in enumerate(ch_names)])
    layout.update(annotations=annotations)

    # set the size of the figure and plot it
    layout.update(autosize=False, width=1000, height=700)
    fig = Figure(data=Data(traces), layout=layout)
    st.plotly_chart(fig)
    #py.iplot(fig, filename='shared xaxis')

def main():
    st.title('Demonstration of EEG data pipeline')
    st.write("""
             This is a simple app for visualising and analysing EEG data. Start by uploading the .EDF files you want to analyse.
             """)
    
    # 1: Upload EDF files
    edf_file_buffers = st.file_uploader('Upload .EDF files', type='edf', accept_multiple_files=True)
    

    if edf_file_buffers:
        data_folder, file_paths = get_file_paths(edf_file_buffers)
        
        # if 'data_processed' in st.session_state:
        #     st.write("Data has been processed")
        #     df = st.session_state.data
        #     start_time = st.slider('Select start time for plot', min_value=st.session_state.time_min, max_value=st.session_state.time_max, value=st.session_state.start_time)
        #     st.session_state.start_time = start_time
        #     fig = visualize_plot_from_eeg_data(st.session_state.data, st.session_state.start_time, DURATION)
        #     st.pyplot(fig)

        # elif st.button("Process data"):
        if st.button("Process data"):
            st.session_state.data_processed = True
            st.write("Data processing initiated")
          
            # # 2: Chop the .edf data into 5 second windows

            # Change annnotation dictionary

            data_dict = load_data_dict(data_folder_path=data_folder, annotation_dict=tuh_eeg_artefact_annotations, stop_after=None, duration=DURATION)
            all_subjects = list(data_dict.keys())
            print(all_subjects)
            print(data_dict)
            X, _ = get_data(data_dict, all_subjects)

            # # 3: Load the model and generate latent representations
            encoder = load_model(device)   
            latent_representations = generate_latent_representations(X, encoder, device=device)

            print(latent_representations)

            classification = classify(latent_representations)

            print(classification)


            output = []
            for w_idx, window in enumerate(classification):
                for l_idx, label in enumerate(window):
                    if label == 1:
                        output.append((w_idx, l_idx))
            print(output)
            for w_idx, l_idx in output:
                time = w_idx * DURATION
                st.write(f"Between {time} and {time + DURATION} we suspect {tuh_eeg_index_to_articfact_annotations[l_idx]}")

            # Visualize
            data = mne.io.read_raw_edf(file_paths[0], preload=True)
            # df = data.to_data_frame()
            # st.session_state.data = df
            def draw_plot():
                # fig = visualize_plot_from_eeg_data(df, st.session_state.slider, DURATION)
                # st.pyplot(fig)
                plot_raw(data, st, st.session_state.slider, st.session_state.slider + DURATION)
                st.slider('Select start time for plot', 
                            min_value=st.session_state.time_min, 
                            max_value=st.session_state.time_max, 
                            value=time_min,
                            key='slider',
                            on_change=draw_plot)
                for w_idx, l_idx in output:
                    time = w_idx * DURATION // 2
                    st.write(f"Between {time} and {time + DURATION} we suspect {tuh_eeg_index_to_articfact_annotations[l_idx]}")
                    st.button(label=f"{tuh_eeg_index_to_articfact_annotations[l_idx]}: {time} - {time + DURATION}",
                                on_click=change_time,
                                args=(time,))
            def change_time(time):
                st.session_state.slider = time
                draw_plot()
            print(file_paths[0], data)
            time_min = 0
            time_max = int(data.times[-1])
            st.session_state.time_min = time_min
            st.session_state.time_max = time_max
            st.session_state.slider = time_min
            draw_plot()
            # st.session_state.start_time = start_time
            # Placeholder for the plot
            # plot_placeholder = st.empty()


            # # 4: Perform KMeans clustering on the latent representations
            # st.write("Running K-means with n=5 clusters")
            # kmeans = KMeans(n_clusters=5, random_state=42)
            # kmeans.fit(latent_representations)
            # labels = kmeans.labels_

            # # 5: Visualize the clusters using PCA 
            # st.write("Visualising clusters using PCA")  
            # # Apply PCA
            # pca = PCA(n_components=2)
            # components = pca.fit_transform(latent_representations)

            # # Plot clusters
            # plot_clusters(components, labels)



main()
