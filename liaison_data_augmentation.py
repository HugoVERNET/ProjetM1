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
import random  # Ajout pour l'augmentation aléatoire

# --- Générer et Sauvegarder un Graphe ---
def generate_and_save_graph(filepath, patient_id, patient_labels, output_dir, use_edge_weights=False):
    y = patient_labels.get(patient_id, torch.tensor([-1, -1], dtype=torch.long))
    df_data = pd.read_csv(filepath)
    print(f"Taille : {df_data.shape[0]} lignes, {df_data.shape[1]} colonnes")

    if np.any(pd.isna(df_data)):
        print(f"Ignoré {filepath} : Contient des valeurs NaN")
        return None

    features = df_data.values
    x = torch.tensor(features, dtype=torch.float)
    print(f"Calcul des distances pour {os.path.basename(filepath)}")
    dist_matrix = euclidean_distances(x.numpy())
    
    num_nodes = x.shape[0]
    threshold = np.percentile(dist_matrix, 10)
    edge_index = []
    edge_weight = [] if use_edge_weights else None
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = dist_matrix[i][j]
            if dist < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                if use_edge_weights:
                    weight = 1.0 / (dist + 1e-8)
                    edge_weight.append(weight)
                    edge_weight.append(weight)

    if not edge_index:
        edge_index = [[0, 1], [1, 0]] if num_nodes > 1 else [[0, 0]]
        if use_edge_weights:
            edge_weight = [1.0, 1.0] if num_nodes > 1 else [1.0]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if use_edge_weights and edge_weight:
        edge_weight = np.array(edge_weight)
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    if y[0] != -1 or y[1] != -1:
        graph = Data(x=x, edge_index=edge_index, y=y)
        if use_edge_weights:
            graph.edge_weight = edge_weight
        graph_path = os.path.join(output_dir, f"{patient_id}_{os.path.basename(filepath).replace('.csv', '')}.pt")
        torch.save(graph, graph_path)
        print(f"Graphe créé et sauvegardé : {graph_path}")
        return graph_path
    return None

# --- Charger et Préparer les Données ---
def prepare_graphs(cluster_dir, rtmlpa_file, clinical_file, output_dir, use_edge_weights=False):
    df_rtmlpa = pd.read_csv(rtmlpa_file)
    df_clinical = pd.read_csv(clinical_file)
    
    df_clinical = df_clinical.dropna(subset=['N° patient'])
    df_clinical['N° patient'] = df_clinical['N° patient'].astype(float).astype(int)
    df = pd.merge(df_rtmlpa, df_clinical, left_on='patient_id', right_on='N° patient', how='inner')
    
    patient_labels = {}
    for _, row in df.iterrows():
        patient_id = str(row['patient_id']).zfill(3)
        response = row['Reponse 1ere ligne']
        label = row['label']
        posneg = -1
        try:
            response = float(response)
            if response in [1, 2, 3]:
                posneg = 1
            elif response in [4, 5]:
                posneg = 0
        except (ValueError, TypeError):
            pass
        subtype = -1
        if label == 'ABC':
            subtype = 0
        elif label == 'GCB':
            subtype = 1
        patient_labels[patient_id] = torch.tensor([posneg, subtype], dtype=torch.long)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
                print(f"Chargement : {filepath}")
                graph_path = generate_and_save_graph(filepath, patient_id, patient_labels, output_dir, use_edge_weights)
                if graph_path:
                    graph_paths.append(graph_path)
                    filename_to_graph[filepath] = graph_path

    print(f"Créé {len(graph_paths)} graphes à partir de {cluster_dir}")
    return graph_paths, filename_to_graph

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

# --- Fonctions d'Augmentation ---
def perturb_features(data, noise_std=0.01):
    data_aug = data.__class__.from_dict(data.to_dict())
    noise = torch.randn_like(data.x) * noise_std
    data_aug.x = data.x + noise
    return data_aug

def drop_edges(data, drop_prob=0.2):
    from torch_geometric.utils import dropout_edge
    edge_index, _ = dropout_edge(data.edge_index, p=drop_prob, force_undirected=True)
    return Data(x=data.x, edge_index=edge_index, y=data.y)

def drop_nodes(data, drop_prob=0.1):
    node_mask = torch.rand(data.num_nodes) >= drop_prob
    x = data.x[node_mask]
    old_to_new = -torch.ones(data.num_nodes, dtype=torch.long)
    old_to_new[node_mask] = torch.arange(node_mask.sum())
    src, dst = data.edge_index
    keep_edge = node_mask[src] & node_mask[dst]
    edge_index = data.edge_index[:, keep_edge]
    edge_index = old_to_new[edge_index]
    return Data(x=x, edge_index=edge_index, y=data.y)

augment_fns = [perturb_features, drop_edges, drop_nodes]

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

# --- Custom Dataset avec Augmentation ---
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_paths, augment=False):
        self.graph_paths = graph_paths
        self.augment = augment

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        print(f"Chargement du graphe : {self.graph_paths[idx]}")
        data = torch.load(self.graph_paths[idx])
        if self.augment:
            aug_fn = random.choice(augment_fns)  # Choisir une méthode aléatoire
            data = aug_fn(data)
        return data

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
    parser.add_argument('--num_epochs', type=int, default=3, help='Nombre d’époques d’entraînement')
    parser.add_argument('--batch_size', type=int, default=8, help='Taille du lot')
    parser.add_argument('--lr', type=float, default=0.001, help='Taux d’apprentissage')
    parser.add_argument('--train_size', type=float, default=0.7, help='Proportion de l’ensemble d’entraînement')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion de l’ensemble de validation')
    parser.add_argument('--output_dir', type=str, default='./models', help='Répertoire pour sauvegarder les modèles entraînés')
    parser.add_argument('--checkpoint', type=str, help='Chemin vers le point de contrôle pour l’inférence')
    parser.add_argument('--regenerate_graphs', action='store_true', help='Régénérer les graphes au lieu de charger les existants')
    parser.add_argument('--augment', action='store_true', help='Activer l’augmentation aléatoire des données')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du périphérique : {device}")

    graph_dir = './graphs_temp'
    if args.regenerate_graphs:
        graph_paths, filename_to_graph = prepare_graphs(args.cluster_dir, args.rtmlpa_file, args.clinical_file, graph_dir, use_edge_weights=False)
    else:
        graph_paths, filename_to_graph = load_existing_graphs(args.cluster_dir, graph_dir)
    if not graph_paths:
        print("Aucun graphe valide chargé ou créé. Sortie.")
        return

    input_dim = torch.load(graph_paths[0]).x.shape[1]
    model = GNN(input_dim=input_dim, size=args.size).to(device)

    if args.mode == 'train':
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        train_graphs, val_graphs, test_graphs = split_data(graph_paths, filename_to_graph, args.train_size, args.val_size)
        # Activer l'augmentation uniquement pour l'ensemble d'entraînement si --augment est spécifié
        train_dataset = GraphDataset(train_graphs, augment=args.augment)
        val_dataset = GraphDataset(val_graphs, augment=False)  # Pas d'augmentation pour validation
        test_dataset = GraphDataset(test_graphs, augment=False)  # Pas d'augmentation pour test

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion_posneg = nn.BCEWithLogitsLoss()
        criterion_subtype = nn.CrossEntropyLoss()

        model_name = f"GCN_NO_WEIGHTS_{args.size.upper()}_{args.data_mode.upper()}"
        if args.augment:
            model_name += "_AUGMENTED"
            print("Augmentation aléatoire activée pour l'ensemble d'entraînement.")
        
        train(model, train_loader, val_loader, optimizer, criterion_posneg, criterion_subtype, 
              device, args.num_epochs, model_name, args.output_dir)

        test_loss, test_acc_posneg, test_acc_subtype = compute_metrics(model, test_loader, criterion_posneg, criterion_subtype, device)
        print(f"\nRésultats des tests pour {model_name}:")
        print(f"Perte Test : {test_loss:.4f}")
        print(f"Précision Posneg Test : {test_acc_posneg:.4f}")
        print(f"Précision Subtype Test : {test_acc_subtype:.4f}")

if __name__ == "__main__":
    main()