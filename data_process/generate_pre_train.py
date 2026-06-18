"""Generate pre-training triple data from partitioned alert subgraphs.

Reads a graph CSV (produced by log2graph.py), extracts base pattern profiles,
and creates (base, match, no_match) triples for self-supervised pre-training
using both pattern matching and graph kernel similarity.

Usage:
    python generate_pre_train.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

import json
import random

import pandas as pd
from sklearn.model_selection import train_test_split
from grakel import Graph, GraphKernel
import tqdm

from src.evaluation.get_base_pattern import get_base_pattern
from config import GRAPH_DATA_DIR, PRE_TRAIN_DIR


def transfomer_graph(input_graph):
    """Convert a graph row to a grakel Graph object."""
    node_attr = json.loads(input_graph['node_attr'])
    edge_attr = json.loads(input_graph['edge_attr'])
    edges = json.loads(input_graph['edge_index'])

    node_labels = {}
    node_id = 0
    for one_node_attr in node_attr:
        ip_is_center, ip_label = one_node_attr[0], one_node_attr[2:]
        ip_label_parse = ip_label.index(1)
        node_labels[node_id] = ip_label_parse
        node_id += 1

    edges_label = {}
    for i, attr in enumerate(edge_attr):
        protocol_len = 20
        protocol_end = 627 + protocol_len
        dprot, protocol, signature, warn_num = (
            attr[0:627], attr[627:protocol_end], attr[protocol_end:-2], attr[-1]
        )
        signature_num = signature.index(1)
        signature_num_str = str(signature_num)
        edge = tuple(edges[i])
        if edge not in edges_label:
            edges_label[edge] = set(signature_num_str)
        else:
            edges_label[edge].add(signature_num_str)

    G = Graph(edges_label, edge_labels=edges_label, node_labels=node_labels)
    return G


def generate_train_data(input_path, output_path):
    """Generate triple training data from graph CSV."""
    graph_train_df = pd.read_csv(input_path)
    print(f"Input graph count: {len(graph_train_df)}")

    if len(graph_train_df) < 5:
        print("Too few graphs (< 5), generating empty output")
        pd.DataFrame(columns=['base_id', 'match_id', 'no_match_id']).to_csv(
            output_path, index=False
        )
        return

    # Compute base patterns
    graph_train_df['base_pattern'] = graph_train_df.apply(
        lambda x: str(get_base_pattern(x)), axis=1
    )

    # Split: 20% base, 80% search
    base_data, search_data = train_test_split(
        graph_train_df, test_size=0.8, random_state=2
    )
    base_data.reset_index(drop=True, inplace=True)
    print(f"Base set: {len(base_data)}, Search set: {len(search_data)}")

    # ── Pattern-based triples ───────────────────────────────────────────────
    search_match = search_data[['id', 'base_pattern']].reset_index(drop=True)
    search_match.rename(columns={'id': 'search_id'}, inplace=True)

    merged = pd.merge(base_data, search_match, on=['base_pattern'], how='left')
    merged['search_id'] = merged['search_id'].fillna(-1).astype(int)

    pattern_match = (
        merged.groupby(['id', 'base_pattern'])['search_id']
        .apply(list)
        .reset_index()
        .rename(columns={'search_id': 'match_id_list'})
    )

    pattern_match['no_matched'] = pattern_match.apply(
        lambda row: row['match_id_list'] == [-1], axis=1
    )
    no_matched_data = pattern_match[pattern_match['no_matched']]
    print(f"No same-pattern match: {len(no_matched_data) / len(base_data) * 100:.1f}%")
    pattern_match.drop(labels=['no_matched'], axis=1, inplace=True)

    pattern_match_data = []
    for i in range(len(pattern_match)):
        base_id = pattern_match.loc[i, 'id']
        pattern_match_ids = pattern_match.loc[i, 'match_id_list']
        match_ids = pattern_match.loc[i, 'match_id_list']

        no_match_ids = search_data[
            ~search_data['id'].isin(pattern_match_ids)
        ]['id'].tolist()
        no_match_2id = random.sample(no_match_ids, min(2, len(no_match_ids)))
        if len(no_match_2id) < 2:
            continue

        if len(match_ids) == 1:
            if match_ids == [-1]:
                continue
            match_id = match_ids[0]
            pattern_match_data.append(
                {'base_id': base_id, 'match_id': match_id, 'no_match_id': no_match_2id[0]}
            )
            pattern_match_data.append(
                {'base_id': base_id, 'match_id': match_id, 'no_match_id': no_match_2id[1]}
            )
        else:
            match_2id = random.sample(match_ids, min(2, len(match_ids)))
            if len(match_2id) < 2:
                continue
            for m_id in match_2id:
                for n_id in no_match_2id:
                    pattern_match_data.append(
                        {'base_id': base_id, 'match_id': m_id, 'no_match_id': n_id}
                    )

    pattern_match_df = pd.DataFrame(pattern_match_data)
    print(f"Pattern-based triples: {len(pattern_match_df)}")

    # ── Graph-kernel-based triples ──────────────────────────────────────────
    aggregated = (
        pattern_match.groupby('base_pattern')
        .agg({'id': list, 'match_id_list': list})
        .reset_index()
    )

    simi_all = []
    graph_kernel_match = []
    num_triplets = 3
    simi_threshold = 0.80
    no_simi_threshold = 0.40

    for i in tqdm.tqdm(range(len(aggregated)), desc="Graph kernel"):
        match_all_list = []
        for now_match_list in aggregated.iloc[i]['match_id_list']:
            match_all_list = match_all_list + now_match_list
        pattern_id_list = match_all_list + aggregated.iloc[i]['id']
        pattern_id_set = set(pattern_id_list)

        Graph_list = []
        graph_id_list = []
        for graph_id in pattern_id_set:
            if graph_id != -1:
                graph = graph_train_df.loc[graph_train_df['id'] == graph_id].iloc[0]
                Graph_a = transfomer_graph(graph)
                Graph_list.append(Graph_a)
                graph_id_list.append(graph_id)

        if len(Graph_list) < 2:
            continue

        sp_kernel = GraphKernel(kernel="NSPD", normalize=True)
        try:
            simi_matrix = sp_kernel.fit_transform(Graph_list)
        except Exception:
            continue

        for i_idx in range(len(simi_matrix)):
            base_id = graph_id_list[i_idx]
            match_id_list = []
            no_match_id_list = []
            triplets = set()
            for j_idx in range(i_idx + 1, len(simi_matrix)):
                sim_now = simi_matrix[i_idx][j_idx]
                simi_all.append(sim_now)
                if sim_now >= simi_threshold:
                    match_id_list.append(graph_id_list[j_idx])
                elif sim_now <= no_simi_threshold:
                    no_match_id_list.append(graph_id_list[j_idx])

            max_possible = len(match_id_list) * len(no_match_id_list)
            if max_possible == 0:
                continue
            limit = min(num_triplets, max_possible)
            while len(triplets) < limit:
                match_id = random.choice(match_id_list)
                no_match_id = random.choice(no_match_id_list)
                triplets.add((base_id, match_id, no_match_id))

            for triplet in triplets:
                base_id_t, match_id_t, no_match_id_t = triplet
                graph_kernel_match.append({
                    'base_id': base_id_t,
                    'match_id': match_id_t,
                    'no_match_id': no_match_id_t,
                })

    graph_kernel_match_df = pd.DataFrame(graph_kernel_match)
    print(f"Graph-kernel triples: {len(graph_kernel_match_df)}")

    # Merge and save
    result_df = pd.concat([pattern_match_df, graph_kernel_match_df], ignore_index=True)
    print(f"Total triples: {len(result_df)}")
    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    input_path = str(GRAPH_DATA_DIR / "correct" / "graph_train.csv")
    output_path = str(PRE_TRAIN_DIR / "trple-graph-pre-all.csv")
    PRE_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    generate_train_data(input_path, output_path)
