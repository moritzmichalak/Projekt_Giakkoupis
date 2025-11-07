import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import copy

# ======
# Create different types of cuts.
# ======

def generate_cut_1(G):
    ''' Generates cut including every 2nd clique: Choose all nodes whose Clique-ID (before '_') is pair '''
    S = set()
    for node in G.nodes():
        clique_id = int(str(node).split('_')[0])  
        if clique_id % 2 == 0:
            S.add(node)
    return S

def generate_cut_2(G, amount_cliques: int, p_clique_size: int):
    ''' Generates smallest Cut, including only entire cliques '''
    S = set()
    nodes = list(G.nodes)
    for i in range(amount_cliques*p_clique_size):
        S.add(nodes[i])
    return S

def generate_cut_3(G, amount_cliques: int, p_clique_size: int):
    ''' Generates big Cut, including half of each clique's nodes '''
    S = set()
    nodes = list(G.nodes)
    nodes_clique_S = p_clique_size // 2
    # for each clique
    for i in range(amount_cliques):
        # add half of its nodes
        for j in range(nodes_clique_S):
            if j < (p_clique_size):
                # rotation
                S.add(f"{i}_{(i+j) % (p_clique_size)}")
    return S

# ======
# Create random graphs:
# ======

def create_random_d_regular_graph(seed=None):
    n = random.randint(10, 30)
    rng = random.Random(seed)
    possible_d = [d for d in range(3, n // 2 + 1) if (n * d) % 2 == 0]
    d = rng.choice(possible_d)
    G = nx.random_regular_graph(d, n, seed=seed)
    pos = nx.spring_layout(G, seed=42) 
    return G, pos

def create_random_even_cycle_graph(seed=None):
    min_n, max_n = 8, 30
    rng = random.Random(seed)
    possible_n = [num for num in range(min_n, max_n + 1) if num % 2 == 0]
    n = rng.choice(possible_n)
    G = nx.cycle_graph(n)
    pos = nx.circular_layout(G)  
    return G, pos, n

def generate_random_cut(G, seed=None):
    nodes = list(G.nodes())
    rng = random.Random(seed)
    max_size = len(nodes) // 2
    size = rng.randint(2, max_size)
    S = set(rng.sample(nodes, size))
    return S

def create_ring_of_cliques(p_num_cliques: int, p_clique_size: int):
    ''' Generates ring of cliques of any desired size. '''
    num_cliques = p_num_cliques
    clique_size = p_clique_size
    d = clique_size - 1
    G = nx.Graph()
    cliques = []
    edges_to_remove = []
    edges_to_add = []
    for i in range(num_cliques):
        nodes = []
        for j in range(clique_size):
            nodes.append(f"{i}_{j}")
        cliques.append(nodes)
        # Add one edge for each pair of nodes so that clique is completely connected.
        # gets all pairs of nodes
        for u, v in itertools.combinations(nodes, 2):
            G.add_edge(u, v)
        # Collect for each clique i the edge ("i_0", "i_1") to be removed.
        edges_to_remove.append((f"{i}_{0}", f"{i}_{d}"))
    pos = {}
    angle_step = 2 * np.pi / num_cliques
    for i, clique in enumerate(cliques):
        angle = i * angle_step
        center_x, center_y = np.cos(angle), np.sin(angle)
        for j, node in enumerate(clique):
            offset_x = 0.3 * np.cos(2 * np.pi * j / clique_size)
            offset_y = 0.3 * np.sin(2 * np.pi * j / clique_size)
            pos[node] = (center_x + offset_x, center_y + offset_y)

    for i in range(num_cliques):
        if i <= num_cliques - 2:
            edges_to_add.append((f"{i}_{0}", f"{i+1}_{d}"))
        else:
            edges_to_add.append((f"{i}_{0}", f"{0}_{d}"))
    for u, v in edges_to_remove:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    for u, v in edges_to_add:
        G.add_edge(u, v)
    return G, pos


def flip_operation(G, num_cliques, size_of_cliques, rng=random):
    ''' Executes flip operation as in Giakkoupis paper. '''
    # 1) Choose a random node based on label
    i = rng.randrange(num_cliques)
    j = rng.randrange(size_of_cliques)
    a = f"{i}_{j}"
    # 2) Choose a random neighbour of a 
    Na_view = G.adj[a]
    if not Na_view: 
        return G, None, None, None
    b = rng.choice(tuple(Na_view))
    # 3) Neighbourhoods obly initially as sets (for effiecency)
    Na = set(Na_view)
    Nb = set(G.adj[b])
    # Possible candidates for b'
    allowed_A = Na - {b} - Nb          # a' ∈ N(a)\{b}\N(b)
    if not allowed_A:
        return G, None, None, None
    allowed_B = Nb - {a} - Na          # b' ∈ N(b)\{a}\N(a)
    if not allowed_B:
        return G, None, None, None
    a_prime = rng.choice(tuple(allowed_A))
    b_prime = rng.choice(tuple(allowed_B))
    # 4) Execute flip operation
    G.remove_edge(a, a_prime)
    G.remove_edge(b, b_prime)
    G.add_edge(a, b_prime)
    G.add_edge(b, a_prime)

    return G, {(a, a_prime), (b, b_prime)}, {(a, b_prime), (b, a_prime)}, (a, b)

def flip_operation_old(G):
    ''' Flip operation as in Schindelhauer/Mahlmann paper '''
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
