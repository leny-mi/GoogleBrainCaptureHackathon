import os
import mne
import numpy as np
from tqdm import tqdm

from src.data.utils.eeg import get_raw

import torch

def normalize_and_add_scaling_channel(x: torch.Tensor, data_min = -0.001, data_max = 0.001, low=-1, high=1, scale_idx=-1):
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    x *= (high - low)

    X = torch.zeros((x.shape[0], x.shape[1] + 1, x.shape[2]))
    X[:, :-1] = x

    max_scale = data_max - data_min

    scale = 2 * (torch.clamp_max((x.max() - x.min()) / max_scale, 1.0) - 0.5)
    X[:, scale_idx] = scale

    return X



def load_data_dict(data_folder_path: str, annotation_dict: dict, duration: float = 1, overlap: float = 0.5, stop_after = None):
    """Loads the data from the data folder.
    Parameters
    ----------
    data_folder_path : str
        The path to the data folder.
    channel_config : list
        The configuration of the channels.
    tmin : float
        The start time.
    tlen : float
        The duration of an epoch.
    labels : bool
        Whether to include labels.
    Returns
    -------
    data_dict : dict
        The data dictionary.
    """
    data_dict = {}
    
    assert(duration == 1)
    assert(overlap == 0.5)

    l = 0
    for subject in tqdm(os.listdir(data_folder_path)):
        l += 1 
        data_dict[subject] = {}

        for session in os.listdir(data_folder_path + subject):
            session_name = session.split('.')[0]
            data_dict[subject][session_name] = {}
            
            edf_file_path = data_folder_path + subject + '/' + session
            raw = get_raw(edf_file_path, filter=True)

            events, _ = mne.events_from_annotations(raw, event_id=annotation_dict, verbose='error')
            epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose='error', overlap=overlap)

            freq = raw.info["sfreq"]

            target_annotations = []
            for i, epoch in enumerate(epochs):
                r = events[((i * (freq//2)) <= events[:,0]) & (events[:,0] < (i * (freq//2) + freq))][:,2]
                target_annotation = [int(j in r) for j in range(len(annotation_dict))]

                target_annotations.append(target_annotation)

            data_dict[subject][session_name]['y'] = torch.tensor(target_annotations)


            X = epochs.get_data().astype(np.float32)

            if X.shape[0] == 0:
                print(f'No epochs in {subject} {session_name}')
                data_dict[subject].pop(session_name)
                continue

            X = normalize_and_add_scaling_channel(torch.tensor(X))

            data_dict[subject][session_name]['X'] = X
        #break
        if stop_after != None and l > stop_after:
            break
    return data_dict


import torch

def get_data(data_dict, subject_list=None):
    """Returns the data and labels.
    Parameters
    ----------
    data_dict : dict
        The data dictionary.
    subject_list : list
        The list of subjects.
    Returns
    -------
    X : torch.Tensor
        The data.
    y : torch.Tensor
        The labels.
    """
    if subject_list is None:
        subject_list = list(data_dict.keys())

    X = [torch.tensor(data_dict[subject][session]['X']) for subject in subject_list for session in data_dict[subject].keys()]
    X = torch.cat(X)

    if 'y' in data_dict[subject_list[0]][list(data_dict[subject_list[0]].keys())[0]]:
        y = [torch.tensor(data_dict[subject][session]['y']) for subject in subject_list for session in data_dict[subject].keys()]
        y = torch.cat(y)
        return X, y

    return X
