
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph  # Utilisé pour générer le graphe k-NN
import os
import numpy as np
import pandas as pd
import argparse
import re
import gc

# --- Générer et Sauvegarder un Graphe avec knn_graph (5 voisins) ---
def generate_and_save_graph(filepath, patient_id, patient_labels, output_dir, use_edge_weights=False):
    y = patient_labels.get(patient_id, torch.tensor([-1, -1], dtype=torch.long))
    df_data = pd.read_csv(filepath)
    print(f"Taille totale : {df_data.shape[0]} lignes, {df_data.shape[1]} colonnes")

    # Exclure la première colonne et prendre les 15 suivantes
    df_data = df_data.iloc[:, 1:16]  # Index 1 à 15 inclus (15 colonnes)
    print(f"Taille après exclusion de la première colonne : {df_data.shape[0]} lignes, {df_data.shape[1]} colonnes")

    if np.any(pd.isna(df_data)):
        print(f"Ignoré {filepath} : Contient des valeurs NaN")
        return None

    features = df_data.values
    x = torch.tensor(features, dtype=torch.float)
    print(f"Génération du graphe k-NN pour {os.path.basename(filepath)}")
    
    # Générer le graphe k-NN avec 5 voisins
    edge_index = knn_graph(x, k=5, batch=None, loop=False, flow='source_to_target')
    
    # Si use_edge_weights est activé, calculer les poids à partir des distances (optionnel)
    edge_weight = None
    if use_edge_weights:
        # Calculer les distances pour les arêtes sélectionnées
        src, dst = edge_index
        dists = torch.norm(x[src] - x[dst], dim=1)
        edge_weight = 1.0 / (dists + 1e-8)
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min() + 1e-8)

    if y[0] != -1 or y[1] != -1:
        graph = Data(x=x, edge_index=edge_index, y=y)
        if use_edge_weights:
            graph.edge_weight = edge_weight
        graph_path = os.path.join(output_dir, f"{patient_id}_{os.path.basename(filepath).replace('.csv', '')}.pt")
        torch.save(graph, graph_path)
        print(f"Graphe créé et sauvegardé : {graph_path}")
        # Nettoyage explicite de la RAM
        del df_data, features, x, edge_index, edge_weight, graph
        gc.collect()
        return graph_path
    return None

# --- Charger et Préparer les Données pour Sauvegarder les Graphes ---
def save_graphs(cluster_dir, rtmlpa_file, clinical_file, output_dir, use_edge_weights=False):
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

    print(f"Créé et sauvegardé {len(graph_paths)} graphes dans {output_dir}")
    return graph_paths

# --- Main avec ArgumentParser ---
def main():
    parser = argparse.ArgumentParser(description="Générer et sauvegarder des graphes à partir de fichiers CSV")
    parser.add_argument('--cluster_dir', type=str, default='./clusters', help='Répertoire contenant les fichiers CSV des clusters')
    parser.add_argument('--rtmlpa_file', type=str, default='RTMLPA.csv', help='Fichier RTMLPA.csv')
    parser.add_argument('--clinical_file', type=str, default='clinical.csv', help='Fichier clinical.csv')
    parser.add_argument('--output_dir', type=str, default='./graphs_temp', help='Répertoire pour sauvegarder les graphes')
    parser.add_argument('--use_edge_weights', action='store_true', help='Utiliser des poids sur les arêtes')

    args = parser.parse_args()

    graph_paths = save_graphs(args.cluster_dir, args.rtmlpa_file, args.clinical_file, args.output_dir, args.use_edge_weights)

if __name__ == "__main__":
    main()
