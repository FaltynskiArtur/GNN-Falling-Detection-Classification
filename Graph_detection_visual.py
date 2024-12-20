import os
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Sprawdzanie dostępności GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ścieżka do przeszukania
path_to_search = r'D:/FIR-Human/FIR-Human'

# Import funkcji przetwarzania danych z osobnego modułu
from data import prepare_data, create_graph, augment_data, extract_data_from_mat

# Import modelu GCN z osobnego modułu
from graph_class import GNNWithResiduals

# Klasy etykiet
class_names = ["Forwards", "Backwards", "Side", "Other"]

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path='best_model.pth', activation_fn=F.relu):
    model = GNNWithResiduals(activation_fn, num_classes=4)  # Obsługa 4 klas
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def process_video(video_path, model, joints_data, class_names):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(joints_data):
            joints = joints_data[frame_idx]
            # Wizualizuj punkty kluczowe na ramce
            for joint in joints:
                x, y, _ = joint
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Konwersja danych punktów kluczowych na graf
        graph = convert_frame_to_graph(joints)
        graph = graph.to(device)

        with torch.no_grad():
            out = model(graph)
            pred = out.argmax(dim=1).item()
            pred_label = class_names[pred]  # Pobierz nazwę klasy

        # Wyświetl przewidywaną klasę na ramce
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Prediction: {pred_label}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Wyświetl ramkę z przewidywaniem i wizualizacją stawów
        cv2.imshow('Video', frame)

        frame_idx += 1

        # Przerwij pętlę, jeśli użytkownik naciśnie klawisz 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def convert_frame_to_graph(joints):
    """
    Konwertuje dane punktów kluczowych na graf.
    """
    num_nodes = joints.shape[0]
    edge_index = torch.tensor([
        [i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j
    ], dtype=torch.long).t().contiguous()
    x = torch.tensor(joints, dtype=torch.float)
    graph = Data(x=x, edge_index=edge_index)
    return graph


if __name__ == "__main__":
    # Wczytaj model
    model = load_model('71Epok99procent.pth', F.relu)

    # Ścieżki do wideo i danych punktów kluczowych
    video_path = r"D:\FIR-Human\FIR-Human\BLOCK_1_TRX\Volunteer_3\3_EATING\FinalVideo.avi"
    joints_data_path = r"D:\FIR-Human\FIR-Human\BLOCK_1_TRX\Volunteer_3\3_EATING\labels.mat"

    # Wczytaj dane punktów kluczowych z pliku .mat
    joints_data = extract_data_from_mat(joints_data_path)

    if joints_data is not None:
        process_video(video_path, model, joints_data, class_names)
    else:
        print("Could not load joints data.")
