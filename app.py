# app.py
import mne
from pdfreport.generatePdf import Report
import streamlit as st
#st.set_page_config(layout="wide")
import sys
import os
sys.path.append(os.getcwd() + '/')
from src.data.processing import load_data_dict, get_data
from src.data.conf.eeg_annotations import  tuh_eeg_artefact_annotations, tuh_eeg_index_to_articfact_annotations
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import torch
from tqdm import tqdm
from copy import deepcopy
from model.model import BendrEncoder
from model.model import Flatten
from sklearn.cluster import KMeans
from src.visualisation.visualisation import plot_latent_pca, visualize_plot_from_eeg_data
import xgboost as xgb
from plotly.graph_objs import Scatter, Layout, YAxis, Annotation, Annotations, Font, Data, Figure

from datetime import date

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
    encoder.eval()

    return encoder

def make_dir(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

def get_file_paths(edf_file_buffer):
    """
    input: edf_file_buffers: list of files uploaded by user

    output: paths: paths to the files
    """
    # make tempoary directory to store the files
    temp_dir = tempfile.mkdtemp()
    
    folder_name = os.path.join(temp_dir, edf_file_buffer.name[:4])
    make_dir(folder_name)
    # make tempoary file
    path = os.path.join(folder_name , edf_file_buffer.name)
    # write bytesIO object to file
    with open(path, 'wb') as f:
        f.write(edf_file_buffer.getvalue())

    return temp_dir + '/', path

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
    start, stop = raw.time_as_index([max(0, range_from - 30), min(raw.times[-1], range_to + 30)])

    n_channels = min(len(picks), 20)
    data, times = raw[picks[:n_channels], start:stop]
    ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]

    step = 1. / n_channels
    kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

    # create objects for layout and traces
    layout = Layout(
        yaxis=YAxis(kwargs), 
        showlegend=False,
        xaxis=dict(
            range=[range_from, range_to]
        )
    )
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
    layout.update(autosize=True)
    fig = Figure(data=Data(traces), layout=layout)
    st.plotly_chart(fig)
    #py.iplot(fig, filename='shared xaxis')

class Observation:
    def __init__(self, window_idx, label_idx, label):
        self.window_idx = window_idx
        self.label_idx = label_idx
        self.label = label
        self.accepted = True
        self.button = None

    def __str__(self):
        return f"{self.label}: {self.start} - {self.end}"
    
    def accept(self):
        self.accepted = True

    def reject(self):
        self.accepted = False


@st.cache_data
def download_custom_pdf(artifacts, annotationInfo):
    report = Report(annotationInfo=annotationInfo, artifacts=artifacts)
    return report().encode("latin-1")

# # 3: Load the model and generate latent representations
encoder = load_model(device)   

@st.cache_data
def process_input(data_folder):
    print("HElloooo", data_folder)
    data_dict = load_data_dict(data_folder_path=data_folder, annotation_dict=tuh_eeg_artefact_annotations, stop_after=None, duration=DURATION)
    all_subjects = list(data_dict.keys())
    X, _ = get_data(data_dict, all_subjects)

    latent_representations = generate_latent_representations(X, encoder, device=device)
    classification = classify(latent_representations)
    
    output = []
    for w_idx, window in enumerate(classification):
        for l_idx, label in enumerate(window):
            if label == 1:
                # output.append((w_idx, l_idx))
                output.append(Observation(w_idx, l_idx, tuh_eeg_index_to_articfact_annotations[l_idx]))

    return tuple(output)

@st.cache_data
def cached_read_raw_edf(file_path):
    return mne.io.read_raw_edf(file_path, preload=True)


def main():
    st.title('Analyzer for bEEGees')
    st.write("""
Analyze EEG recordings to identify regions of interest. Powered by GCP.""")
    
    # 1: Upload EDF files
    edf_file_buffers = st.file_uploader('Upload .EDF files', type='edf', accept_multiple_files=False)
    

    if edf_file_buffers:
        if 'data_folder' not in st.session_state:

            data_folder, file_path = get_file_paths(edf_file_buffers)
            st.session_state.data_folder = data_folder
            st.session_state.file_path = file_path

        st.session_state.data_processed = True

        st.header("Detected Regions of Interest")

        data = cached_read_raw_edf(st.session_state.file_path)

        if 'time_min' not in st.session_state:
            st.session_state.time_min = 0
            st.session_state.time_max = int(data.times[-1])
            st.session_state.slider = 0
        
        col1, col2 = st.columns(2)

        def change_time(time):
            st.session_state.slider = time

        def change_observation_verdict(obs):
            print("Changing observation verdict")
            if obs.accepted:
                obs.reject()
            else:
                obs.accept()

        if 'output' not in st.session_state:
            st.session_state.output = process_input(data_folder=st.session_state.data_folder)

        for obs in st.session_state.output:
            time = obs.window_idx * DURATION // 2
            # with col1:
            col1.button(label=f"{obs.label}: {time} - {time + DURATION}",
                        on_click=change_time,
                        args=(time,))
            # with col2:
            col2.button("Accept" if not obs.accepted else "Reject", key=f"button_{obs.window_idx}_{obs.label_idx}", type='primary', on_click=change_observation_verdict, args=(obs,))


        st.header("EEG Signals")

        st.slider('Select start time for plot', 
            min_value=st.session_state.time_min, 
            max_value=st.session_state.time_max,
            disabled=True,
            label_visibility="collapsed",
            key='slider')
        
        plot_raw(data, st, st.session_state.slider - DURATION // 2, st.session_state.slider + DURATION // 2)


        artifacts = [tuple(map(str, (o.label, o.window_idx * DURATION // 2, (o.window_idx + 1) * DURATION // 2))) for o in st.session_state.output if o.accepted]
        annotationInfo = {
            "annotationDate": date(year=2024, month=3, day=16),
            "annotater": "bEEGees Hackathon Team"
        }

        data = download_custom_pdf(artifacts=artifacts, annotationInfo=annotationInfo)
        st.download_button(label="Download Report", data=data, file_name="EEG_Report.pdf", use_container_width=True)


# Custom CSS to style the buttons
st.markdown("""
<style>
button[kind="primary"] {
    background-color: grey;
}
</style>""", unsafe_allow_html=True)

main()
