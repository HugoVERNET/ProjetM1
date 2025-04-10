import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import re
from collections import defaultdict
import gc

# --- Charger les Graphes Existants ---
def load_existing_graphs(cluster_dir, output_dir):
    graph_paths = []
    filename_to_graph = {}
    for root, _, files in os.walk(cluster_dir):
        for filename in files:
            if filename.endswith('.csv'):
                filepath = os.path.join(root, filename)
                patient_id = os.path.basename(root)
                if not re.match(r'^\d{3}$', patient_id):
                    print(f"Avertissement : Dossier {root} ne correspond pas à un ID patient valide, ignoré")
                    continue
                graph_filename = f"{patient_id}_{os.path.basename(filepath).replace('.csv', '')}.pt"
                graph_path = os.path.join(output_dir, graph_filename)
                if os.path.exists(graph_path):
                    graph_paths.append(graph_path)
                    filename_to_graph[filepath] = graph_path
                else:
                    print(f"Graphe manquant : {graph_path}")
    print(f"Chargé {len(graph_paths)} graphes existants à partir de {output_dir}")
    return graph_paths, filename_to_graph

# --- Data Splitting par Patient ID ---
def split_data(graph_paths, filename_to_graph, train_size=0.7, val_size=0.15):
    patient_to_graphs = defaultdict(list)
    for filepath, graph_path in filename_to_graph.items():
        patient_id = os.path.basename(os.path.dirname(filepath))
        if not re.match(r'^\d{3}$', patient_id):
            print(f"Avertissement : Chemin {filepath} ne correspond pas à un ID patient valide, ignoré")
            continue
        patient_to_graphs[patient_id].append(graph_path)

    patient_ids = list(patient_to_graphs.keys())
    test_size = max(0, 1.0 - (train_size + val_size))
    train_val_ids, test_ids = train_test_split(patient_ids, test_size=test_size, random_state=42)
    relative_val_size = val_size / (train_size + val_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=relative_val_size, random_state=42)

    train_graphs = []
    val_graphs = []
    test_graphs = []
    for pid in train_ids:
        train_graphs.extend(patient_to_graphs[pid])
    for pid in val_ids:
        val_graphs.extend(patient_to_graphs[pid])
    for pid in test_ids:
        test_graphs.extend(patient_to_graphs[pid])

    print(f"Répartition des données : Train={len(train_graphs)} graphes ({len(train_ids)} patients), "
          f"Val={len(val_graphs)} graphes ({len(val_ids)} patients), "
          f"Test={len(test_graphs)} graphes ({len(test_ids)} patients)")
    return train_graphs, val_graphs, test_graphs

# --- Custom Dataset pour Charger les Graphes à la Demande ---
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_paths):
        self.graph_paths = graph_paths

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        print(f"Chargement du graphe : {self.graph_paths[idx]}")
        return torch.load(self.graph_paths[idx])

# --- Modèle GNN ---
class GNN(nn.Module):
    def __init__(self, input_dim, size='petit'):
        super(GNN, self).__init__()
        if size == 'petit':
            hidden_dim = 16
            num_layers = 2
        elif size == 'moyen':
            hidden_dim = 64
            num_layers = 4
        elif size == 'grand':
            hidden_dim = 128
            num_layers = 8
        else:
            raise ValueError("La taille doit être 'petit', 'moyen' ou 'grand'")

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.fc_posneg = nn.Linear(hidden_dim, 1)
        self.fc_subtype = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = global_mean_pool(x, batch)
        posneg_out = self.fc_posneg(x)
        subtype_out = self.fc_subtype(x)
        return posneg_out, subtype_out

# --- Calcul des Métriques ---
def compute_metrics(model, loader, criterion_posneg, criterion_subtype, device):
    model.eval()
    total_loss = 0
    correct_posneg, total_posneg = 0, 0
    correct_subtype, total_subtype = 0, 0

    with torch.no_grad():
        for batch in loader:
            print("Évaluation : Chargement d’un batch")
            batch = batch.to(device)
            posneg_out, subtype_out = model(batch.x, batch.edge_index, batch.batch)

            if batch.y.dim() == 1:
                y_posneg, y_subtype = batch.y[0].float(), batch.y[1]
                batch_size = 1
            else:
                y_posneg, y_subtype = batch.y[:, 0].float(), batch.y[:, 1]
                batch_size = batch.y.size(0)

            mask_posneg = y_posneg != -1
            mask_subtype = y_subtype != -1
            print(f"Évaluation : y_posneg={y_posneg}, mask_posneg={mask_posneg}, y_subtype={y_subtype}, mask_subtype={mask_subtype}")

            loss_posneg = torch.tensor(0.0, device=device)
            if mask_posneg.any():
                posneg_out_masked = posneg_out[mask_posneg].view(-1)
                y_posneg_masked = y_posneg[mask_posneg].view(-1)
                print(f"Évaluation : posneg_out_masked={posneg_out_masked}, y_posneg_masked={y_posneg_masked}")
                loss_posneg = criterion_posneg(posneg_out_masked, y_posneg_masked)

            loss_subtype = torch.tensor(0.0, device=device)
            if mask_subtype.any():
                valid_subtype_targets = y_subtype[mask_subtype]
                if (valid_subtype_targets >= 0).all() and (valid_subtype_targets < 2).all():
                    print(f"Évaluation : subtype_out={subtype_out}, valid_subtype_targets={valid_subtype_targets}")
                    loss_subtype = criterion_subtype(subtype_out, valid_subtype_targets.long())

            loss = loss_posneg + loss_subtype
            total_loss += loss.item() * batch.num_graphs

            if mask_posneg.any():
                pred_posneg = (torch.sigmoid(posneg_out[mask_posneg]) > 0.5).float()
                correct_posneg += (pred_posneg == y_posneg[mask_posneg]).sum().item()
                total_posneg += mask_posneg.sum().item()

            if mask_subtype.any():
                valid_subtype_targets = y_subtype[mask_subtype]
                if (valid_subtype_targets >= 0).all() and (valid_subtype_targets < 2).all():
                    pred_subtype = subtype_out.argmax(dim=-1)[mask_subtype]
                    correct_subtype += (pred_subtype == valid_subtype_targets).sum().item()
                    total_subtype += mask_subtype.sum().item()

            del batch
            gc.collect()

    avg_loss = total_loss / len(loader.dataset)
    acc_posneg = correct_posneg / total_posneg if total_posneg > 0 else 0
    acc_subtype = correct_subtype / total_subtype if total_subtype > 0 else 0
    return avg_loss, acc_posneg, acc_subtype

# --- Boucle d'Entraînement ---
def train(model, train_loader, val_loader, optimizer, criterion_posneg, criterion_subtype, device, num_epochs, model_name, output_dir):
    train_losses, val_losses = [], []
    train_acc_posneg, val_acc_posneg = [], []
    train_acc_subtype, val_acc_subtype = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Début de l’époque {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            print(f"Batch {i+1}/{len(train_loader)} : Chargement")
            batch = batch.to(device)
            print(f"Batch {i+1}/{len(train_loader)} : Calcul du modèle")
            optimizer.zero_grad()
            posneg_out, subtype_out = model(batch.x, batch.edge_index, batch.batch)

            if batch.y.dim() == 1:
                y_posneg, y_subtype = batch.y[0].float(), batch.y[1]
                batch_size = 1
            else:
                y_posneg, y_subtype = batch.y[:, 0].float(), batch.y[:, 1]
                batch_size = batch.y.size(0)

            mask_posneg = y_posneg != -1
            mask_subtype = y_subtype != -1
            print(f"Train : y_posneg={y_posneg}, mask_posneg={mask_posneg}, y_subtype={y_subtype}, mask_subtype={mask_subtype}")

            loss_posneg = torch.tensor(0.0, device=device)
            if mask_posneg.any():
                posneg_out_masked = posneg_out[mask_posneg].view(-1)
                y_posneg_masked = y_posneg[mask_posneg].view(-1)
                print(f"Train : posneg_out_masked={posneg_out_masked}, y_posneg_masked={y_posneg_masked}")
                loss_posneg = criterion_posneg(posneg_out_masked, y_posneg_masked)

            loss_subtype = torch.tensor(0.0, device=device)
            if mask_subtype.any():
                valid_subtype_targets = y_subtype[mask_subtype]
                if (valid_subtype_targets >= 0).all() and (valid_subtype_targets < 2).all():
                    print(f"Train : subtype_out={subtype_out}, valid_subtype_targets={valid_subtype_targets}")
                    loss_subtype = criterion_subtype(subtype_out, valid_subtype_targets.long())

            loss = loss_posneg + loss_subtype
            print(f"Batch {i+1}/{len(train_loader)} : Calcul de la perte = {loss.item()}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

            del batch, posneg_out, subtype_out, loss
            gc.collect()

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        val_loss, val_acc_p, val_acc_s = compute_metrics(model, val_loader, criterion_posneg, criterion_subtype, device)
        _, train_acc_p, train_acc_s = compute_metrics(model, train_loader, criterion_posneg, criterion_subtype, device)

        train_acc_posneg.append(train_acc_p)
        val_acc_posneg.append(val_acc_p)
        train_acc_subtype.append(train_acc_s)
        val_acc_subtype.append(val_acc_s)
        val_losses.append(val_loss)

        print(f'{model_name} - Époque {epoch+1}/{num_epochs}, Perte Entraînement : {train_loss:.4f}, Perte Validation : {val_loss:.4f}, '
              f'Précision Entraînement P/S : {train_acc_p:.4f}/{train_acc_s:.4f}, Précision Validation P/S : {val_acc_p:.4f}/{val_acc_s:.4f}')

        epoch_model_path = os.path.join(output_dir, f'trained_{model_name}_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), epoch_model_path)
        print(f"Modèle sauvegardé à l'époque {epoch+1} sous '{epoch_model_path}'")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, f'trained_{model_name}_best.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"Meilleur modèle mis à jour avec Perte Validation : {val_loss:.4f} à l'époque {epoch+1}")

    plt.figure(figsize=(15, 5))
    for i, (title, train_data, val_data) in enumerate([
        ('Perte', train_losses, val_losses),
        ('Précision Posneg', train_acc_posneg, val_acc_posneg),
        ('Précision Subtype', train_acc_subtype, val_acc_subtype)
    ]):
        plt.subplot(1, 3, i + 1)
        plt.plot(range(1, num_epochs + 1), train_data, 'b-', label=f'Entraînement {title}')
        plt.plot(range(1, num_epochs + 1), val_data, 'r-', label=f'Validation {title}')
        plt.title(f'{model_name} - {title}')
        plt.xlabel('Époque')
        plt.ylabel(title)
        plt.legend()
    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_acc_posneg, val_acc_posneg, train_acc_subtype, val_acc_subtype

# --- Main avec ArgumentParser ---
def main():
    parser = argparse.ArgumentParser(description="Entraîner ou inférer avec un réseau neuronal graphique")
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], default='train', help='Mode : entraîner ou inférer')
    parser.add_argument('--model_type', type=str, choices=['gcn_with_weights', 'gcn_no_weights', 'gat'], 
                        default='gcn_no_weights', help='Type de modèle GNN')
    parser.add_argument('--data_mode', type=str, choices=['clusters', 'cells'], default='clusters', help='Mode des données : clusters ou cellules')
    parser.add_argument('--cluster_dir', type=str, default='./clusters', help='Répertoire contenant les fichiers CSV des clusters')
    parser.add_argument('--cell_dir', type=str, default='./cells', help='Répertoire contenant les fichiers CSV des cellules')
    parser.add_argument('--rtmlpa_file', type=str, default='RTMLPA.csv', help='Fichier RTMLPA.csv')
    parser.add_argument('--clinical_file', type=str, default='clinical.csv', help='Fichier clinical.csv')
    parser.add_argument('--size', type=str, choices=['petit', 'moyen', 'grand'], default='petit', help='Taille du modèle')
    parser.add_argument('--num_epochs', type=int, default=3, help='Nombre d’époques d’entraînement')  # Réduit pour tester
    parser.add_argument('--batch_size', type=int, default=8, help='Taille du lot')
    parser.add_argument('--lr', type=float, default=0.001, help='Taux d’apprentissage')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion de l’ensemble d’entraînement')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion de l’ensemble de validation')
    parser.add_argument('--output_dir', type=str, default='./models', help='Répertoire pour sauvegarder les modèles entraînés')
    parser.add_argument('--checkpoint', type=str, help='Chemin vers le point de contrôle pour l’inférence')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du périphérique : {device}")

    graph_dir = './graphs_temp'
    graph_paths, filename_to_graph = load_existing_graphs(args.cluster_dir, graph_dir)
    if not graph_paths:
        print("Aucun graphe valide chargé. Sortie.")
        return

    input_dim = torch.load(graph_paths[0]).x.shape[1]
    model = GNN(input_dim=input_dim, size=args.size).to(device)

    if args.mode == 'train':
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        train_graphs, val_graphs, test_graphs = split_data(graph_paths, filename_to_graph, args.train_size, args.val_size)
        train_dataset = GraphDataset(train_graphs)
        val_dataset = GraphDataset(val_graphs)
        test_dataset = GraphDataset(test_graphs)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion_posneg = nn.BCEWithLogitsLoss()
        criterion_subtype = nn.CrossEntropyLoss()

        model_name = f"GCN_NO_WEIGHTS_{args.size.upper()}_{args.data_mode.upper()}"
        train(model, train_loader, val_loader, optimizer, criterion_posneg, criterion_subtype, 
              device, args.num_epochs, model_name, args.output_dir)

        test_loss, test_acc_posneg, test_acc_subtype = compute_metrics(model, test_loader, criterion_posneg, criterion_subtype, device)
        print(f"\nRésultats des tests pour {model_name}:")
        print(f"Perte Test : {test_loss:.4f}")
        print(f"Précision Posneg Test : {test_acc_posneg:.4f}")
        print(f"Précision Subtype Test : {test_acc_subtype:.4f}")

if __name__ == "__main__":
    main()

             
