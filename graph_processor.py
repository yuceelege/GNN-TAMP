"""
Graph processing utilities for parsing and converting scene graphs.
"""
import re
import numpy as np
import networkx as nx
import os
import torch
from torch_geometric.data import Data


def read_g_file(filename):
    """Read content from a .g file."""
    with open(filename, 'r') as file:
        content = file.read()
    return content


def get_relative_position(G, start_node, end_node):
    """Get relative position between two nodes in a graph."""
    try:
        edge_data = G.get_edge_data(start_node, end_node)
        if edge_data is not None:
            relative_pos = edge_data['weight']
            return relative_pos
        else:
            print("No direct edge between these nodes.")
            return None
    except KeyError:
        print("One or both of the nodes do not exist in the graph.")
        return None


def parse_transform(transform_string):
    """Parse transformation string and extract translation vector."""
    translation_match = re.search(r't\((.*?)\)', transform_string)
    if translation_match:
        translation_values = translation_match.group(1)
        return np.array([float(val) for val in translation_values.split()])
    return np.zeros(3)


# Precompile regular expressions
position_pattern = re.compile(r'object(\d+).*?X:\s*\[([^\]]+)\]')
transform_pattern = re.compile(r'object(\d+)\(object(\d+)\).*?Q:\s*"([^"]+)"')


def process_g_file(filepath):
    """Process a single .g file and convert it to a graph representation."""
    g_content = read_g_file(filepath)

    object_positions = {}
    object_transformations = {}

    for line in g_content.split('\n'):
        if line.strip() == "":
            continue

        position_match = position_pattern.match(line)
        if position_match:
            obj_id = int(position_match.group(1)) - 1
            position = np.array([float(n) for n in position_match.group(2).split(', ')[:3]])
            object_positions[obj_id] = position

        transform_match = transform_pattern.match(line)
        if transform_match:
            obj_id = int(transform_match.group(1)) - 1
            base_obj_id = int(transform_match.group(2)) - 1
            transform_string = transform_match.group(3)
            transformation = parse_transform(transform_string)
            object_transformations[obj_id] = (base_obj_id, transformation)

    G = nx.DiGraph()

    # Update positions with transformations
    for obj_id, (base_obj_id, transformation) in object_transformations.items():
        if base_obj_id in object_positions:
            transformed_pos = object_positions[base_obj_id] + transformation
            object_positions[obj_id] = transformed_pos

    # Add edges considering the pair only once
    for start_id, start_pos in object_positions.items():
        for end_id, end_pos in object_positions.items():
            if start_id < end_id:  # Ensures each pair is considered only once
                relative_pos = end_pos - start_pos
                if not G.has_edge(end_id, start_id) or not np.array_equal(-relative_pos, G[end_id][start_id]['weight']):
                    G.add_edge(start_id, end_id, weight=relative_pos)
    
    # Ensure edges point upward (remove downward edges)
    for (u, v, w) in list(G.edges(data='weight')):
        if w[2] < 0:
            G.remove_edge(u, v)
            G.add_edge(v, u, weight=-w)
    
    return G, object_positions


def process_all_g_files(directory):
    """Process all .g files in a directory and return graphs and positions."""
    all_graphs = []
    all_positions = []
    for filename in os.listdir(directory):
        if filename.endswith('.g'):
            filepath = os.path.join(directory, filename)
            G, positions = process_g_file(filepath)
            all_graphs.append(G)
            all_positions.append(positions)
    return all_graphs, all_positions


def convert_to_pyg_data(graph):
    """Convert a NetworkX graph to PyTorch Geometric Data format."""
    node_features = torch.ones((graph.number_of_nodes(), 1))
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([graph[u][v]['weight'] for u, v in graph.edges()], dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=0)


def correct_graph_edge_indices(graphs):
    """Correct edge indices in graphs (shift by -1 if needed)."""
    for graph in graphs:
        corrected_edges_with_attrs = []

        # Store edges with corrected indices and their attributes
        for u, v, attrs in graph.edges(data=True):
            corrected_u = u - 1 if u > 0 else u
            corrected_v = v - 1 if v > 0 else v
            corrected_edges_with_attrs.append((corrected_u, corrected_v, attrs))

        # Clear existing edges and re-add them with original attributes
        graph.clear_edges()
        for u, v, attrs in corrected_edges_with_attrs:
            graph.add_edge(u, v, **attrs)

