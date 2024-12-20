# data_processing.py

import os
import numpy as np
import scipy.io
import torch
from torch_geometric.data import Data


# Etykiety kolumn (jeśli potrzebne do innych funkcji)
column_labels = [
    "Head top", "Head front", "LShoulder", "LElbow", "LWrist", "RShoulder",
    "RElbow", "RWrist", "Neck", "Left hip", "Right hip", "Left Knee",
    "Left Ankle", "Left heel", "Left toe", "Right Knee", "Right Ankle",
    "Right heel", "Right toe"
]

# Mapa etykiet (również może być przydatna w innych częściach projektu)
label_map = {
    '1_FORWARDS': 0,
    '2_BACKWARDS': 1,
    '3_SIDE': 2,
    'OTHER': 3  # New label
}

def find_labels_files(path):
    """
    Znajduje pliki labels.mat w podanej ścieżce.
    """
    labels_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file == 'labels.mat':
                labels_files.append(os.path.join(root, file))
    return labels_files

def interpolate_nan(data):
    """
    Interpoluje brakujące wartości (NaN) w danych.
    """
    for i in range(data.shape[1]):
        if np.isnan(data[:, i]).all():
            if i == 0:
                data[:, i] = 0
            else:
                data[:, i] = data[:, i - 1]
        else:
            nan_mask = np.isnan(data[:, i])
            data[nan_mask, i] = np.interp(
                np.flatnonzero(nan_mask),
                np.flatnonzero(~nan_mask),
                data[~nan_mask, i]
            )
    return data

def extract_data_from_mat(file_path):
    """
    Ekstrahuje dane z plików .mat i interpoluje NaNy.
    """
    try:
        mat = scipy.io.loadmat(file_path)
        if 'data_3D' not in mat:
            return None
        data_3D = mat['data_3D']
        if data_3D is None or data_3D.size == 0:
            return None
        for i in range(data_3D.shape[2]):
            data_3D[:, :, i] = interpolate_nan(data_3D[:, :, i])
        return data_3D
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def prepare_data(path_to_search):
    """
    Prepares data for training the model, extracting data from .mat files and assigning labels.
    """
    labels_files = find_labels_files(path_to_search)
    if not labels_files:
        print("No labels.mat files found.")
        return []

    all_data = []
    for file in labels_files:
        data = extract_data_from_mat(file)
        if data is not None:
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file))))

            # Attempt to retrieve label from label_map, default to 'OTHER' if not found
            label = label_map.get(folder_name, label_map['OTHER'])

            if label != -1:  # Ensures valid label
                all_data.append((data, label))
    if not all_data:
        print("No valid data extracted from .mat files.")
    return all_data

def create_graph(data, label):
    """
    Tworzy grafy z danych o pozycji 3D.
    """
    num_nodes = data.shape[1]
    edge_index = torch.tensor([
        [i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j
    ], dtype=torch.long).t().contiguous()

    x = torch.tensor(data.reshape(-1, num_nodes, 3), dtype=torch.float)
    graphs = [Data(x=x[i], edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)) for i in range(x.shape[0])]

    return graphs

def augment_data(graphs):
    """
    Dokonuje perturbacji krawędzi grafów w celu ich augmentacji.
    """
    def perturb_edges(data, prob=0.1):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > prob
        data.edge_index = edge_index[:, mask]
        return data

    return [perturb_edges(graph) for graph in graphs]
