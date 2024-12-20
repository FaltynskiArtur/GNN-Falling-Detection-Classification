import os
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Update model import
from graph_class import GNNWithResiduals  # The model now supports 4 classes
# Update data processing import
from data import prepare_data, create_graph, augment_data, extract_data_from_mat

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path for data
path_to_search = 'D:/FIR-Human/FIR-Human'

def train(model, train_loader, val_loader, optimizer, criterion, early_stopping_patience=10):
    patience_counter = 0
    best_val_acc = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    train_losses, train_accuracies = [], []

    for epoch in range(1, 2):
        model.train()
        total_loss = 0
        correct = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_acc, val_loss, val_y_true, val_y_pred = test(model, val_loader, criterion)
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_model(model, path='best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    plot_metrics(train_losses, train_accuracies)
    plot_confusion_matrix(val_y_true, val_y_pred, class_names=['Forwards', 'Backwards', 'Side', 'Other'])

def test(model, loader, criterion=None):
    model.eval()
    correct = 0
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if criterion:
                loss = criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    accuracy = correct / len(loader.dataset)
    if criterion:
        loss = total_loss / len(loader.dataset)
        return accuracy, loss, y_true, y_pred  # Zwraca 4 wartości
    return accuracy, y_true, y_pred  # Zwraca 3 wartości


def plot_metrics(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path='model.pth', activation_fn=F.relu):
    model = GNNWithResiduals(activation_fn, num_classes=4)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

def process_new_data(new_data_path, model):
    """
    Przetwarza nowe dane w formacie grafów i używa modelu do przewidywania wyników.

    Args:
        new_data_path (str): Ścieżka do nowych danych (np. plik .mat).
        model (torch.nn.Module): Wytrenowany model GCN.
    """
    # Wczytanie danych z pliku .mat
    data = extract_data_from_mat(new_data_path)
    if data is None:
        print(f"Could not load data from {new_data_path}")
        return

    # Przetworzenie danych na grafy
    graphs = []
    for i in range(data.shape[2]):  # Iteracja przez próbki
        graph = create_graph(data[:, :, i], label=None)  # Brak etykiety w nowych danych
        graphs.extend(graph)

    # Przewidywanie wyników
    predictions = []
    model.eval()
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            out = model(graph)
            pred = out.argmax(dim=1).item()
            predictions.append(pred)

    # Wyświetlenie wyników
    for i, pred in enumerate(predictions):
        print(f"Sample {i + 1}: Predicted class = {pred}")


def main(train_model=True, use_model=False, new_data_path=None, activation_fn=F.relu):
    if train_model:
        # Przygotowanie danych
        data = prepare_data(path_to_search)
        if not data:
            print("No data to train the model.")
            return

        # Konwersja do grafów
        all_graphs = []
        for single_data, label in data:
            all_graphs.extend(create_graph(single_data, label))

        # Augmentacja danych
        augmented_graphs = augment_data(all_graphs)

        # Podział na zbiór treningowy, walidacyjny i testowy
        train_graphs, temp_graphs = train_test_split(augmented_graphs, test_size=0.3, random_state=42)
        val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

        print(f"Training graphs: {len(train_graphs)}, Validation graphs: {len(val_graphs)}, Test graphs: {len(test_graphs)}")

        # Model
        model = GNNWithResiduals(activation_fn, num_classes=4)
        model.to(device)

        # Optymalizator i funkcja straty
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        class_weights = torch.tensor([1.0, 2.0, 3.0, 1.0], device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Trenowanie modelu
        train(
            model,
            DataLoader(train_graphs, batch_size=32, shuffle=True),
            DataLoader(val_graphs, batch_size=32, shuffle=False),
            optimizer,
            criterion,
        )

        # Ewaluacja na zbiorze testowym
        test_result = test(
            model, DataLoader(test_graphs, batch_size=32, shuffle=False), criterion=None
        )

        if len(test_result) == 4:
            test_acc, _, test_y_true, test_y_pred = test_result
        else:
            test_acc, test_y_true, test_y_pred = test_result
        print(f"Test Accuracy: {test_acc:.4f}")

        # Wyświetlenie macierzy pomyłek dla zbioru testowego
        plot_confusion_matrix(
            test_y_true, test_y_pred, class_names=["Forwards", "Backwards", "Side", "Other"]
        )

    if use_model and new_data_path is not None:
        model = load_model('best_model.pth', activation_fn)
        process_new_data(new_data_path, model)

if __name__ == "__main__":
    main(train_model=True, use_model=False, new_data_path='labels.mat', activation_fn=F.relu)
