import networkx as nx  # oder: import networkx as xn
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import copy


def create_random_d_regular_graph(seed=None):
    n = random.randint(10, 30)
    rng = random.Random(seed)
    possible_d = [d for d in range(3, n // 2 + 1) if (n * d) % 2 == 0]
    d = rng.choice(possible_d)
    # d = random.randint(2, n/2)
    G = nx.random_regular_graph(d, n, seed=seed)
    pos = nx.spring_layout(G, seed=42)  # feste Knotenpositionen
    return G, pos

def create_random_even_cycle_graph(seed=None):
    # d = 2
    min_n, max_n = 8, 30 
    rng = random.Random(seed)
    #Da d = 2 und d * n gerade sein muss, wähle n gerade: 
    possible_n = [num for num in range(min_n, max_n + 1) if num % 2 == 0]
    n = rng.choice(possible_n)
    G = nx.cycle_graph(n)
    pos = nx.circular_layout(G)  # Fixiertes Kreis-Layout
    return G, pos, n

def create_ring_of_cliques(p_num_cliques: int, p_clique_size: int):
    num_cliques = p_num_cliques
    clique_size = p_clique_size
    G = nx.Graph()
    cliques = []

    for i in range(num_cliques):
        nodes = [f"C{i}_{j}" for j in range(clique_size)]
        G.add_nodes_from(nodes)
        for u, v in itertools.combinations(nodes, 2):
            G.add_edge(u, v)
        cliques.append(nodes)

    pos = {}
    angle_step = 2 * np.pi / num_cliques
    for i, clique in enumerate(cliques):
        angle = i * angle_step
        center_x, center_y = np.cos(angle), np.sin(angle)
        for j, node in enumerate(clique):
            offset_x = 0.3 * np.cos(2 * np.pi * j / clique_size)
            offset_y = 0.3 * np.sin(2 * np.pi * j / clique_size)
            pos[node] = (center_x + offset_x, center_y + offset_y)

    edges_to_remove = [
        ("C2_4", "C2_0"),
        ("C1_3", "C1_4"),
        ("C0_2", "C0_3"),
        ("C4_2", "C4_1"),
        ("C3_1", "C3_0"),
    ]
    for u, v in edges_to_remove:
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    edges_to_add = [
        ("C2_0", "C1_3"),
        ("C1_4", "C0_2"),
        ("C4_1", "C0_3"),
        ("C3_0", "C4_2"),
        ("C2_4", "C3_1"),
    ]
    for u, v in edges_to_add:
        G.add_edge(u, v)

    return G, pos


def flip_operation(G):
    G = copy.deepcopy(G)
    a, b = random.choice(list(G.edges))
    possible_c = list(set(G.neighbors(b)) - {a})
    if not possible_c:
        return G, None, None
    c = random.choice(possible_c)
    possible_d = list(set(G.neighbors(c)) - {a, b})
    if not possible_d:
        return G, None, None
    d = random.choice(possible_d)
    if G.has_edge(a, c) or G.has_edge(b, d):
        return G, None, None
    G.add_edge(a, c)
    G.add_edge(b, d)
    G.remove_edge(a, b)
    G.remove_edge(c, d)

    return G, {(a, b), (c, d)}, {(a, c), (b, d)}


def generate_random_cut(G, seed=None):
    nodes = list(G.nodes())
    rng = random.Random(seed)
    max_size = len(nodes) // 2
    size = rng.randint(2, max_size)
    S = set(rng.sample(nodes, size))
    return S
