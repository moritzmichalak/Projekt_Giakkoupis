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

def generate_cut_1(G):
    # Jede zweite Clique: wähle alle Knoten, deren Clique-ID (vor dem '_') gerade ist
    S = set()
    for node in G.nodes():
        clique_id = int(str(node).split('_')[0])  # <-- korrektes Parsen
        if clique_id % 2 == 0:
            S.add(node)
    return S
'''
def generate_cut_1(G):
    S_ = [] # hässlicher Code, zu optimieren
    nodes = list(G.nodes)
    for i in range(len(nodes)):
        if int(nodes[i][0]) % 2 == 0:
            S_.append(nodes[i])
    S =  {S_[i] for i in range(len(S_))}
    return S
'''
def generate_cut_2(G, amount_cliques: int, p_clique_size: int):
    S= set()
    # S_ = [] # hässlicher Code, zu optimieren
    nodes = list(G.nodes)
    for i in range(amount_cliques*p_clique_size):
        S.add(nodes[i])
        #S_.append(nodes[i])
    # S =  {S_[i] for i in range(len(S_))}
    return S

# Maximum Cut Size:
def generate_cut_3(G, amount_cliques: int, p_clique_size: int):
    # S_ = [] # hässlicher Code, zu optimieren
    S = set()
    nodes = list(G.nodes)
    nodes_clique_S = p_clique_size // 2
    # Für jede Clique:
    for i in range(amount_cliques):
        # Füge die Hälfte der Knoten zu S hinzu: 
        for j in range(nodes_clique_S):
            # print(f"{i}","I")
            if j < (p_clique_size) :
                # S_.append(f"{i}_{(i+j) % (p_clique_size)}")
                S.add(f"{i}_{(i+j) % (p_clique_size)}")
                # print(f"{i}_{j}")
    #S =  {S_[i] for i in range(len(S_))}
    return S

def generate_random_cut(G, seed=None):
    nodes = list(G.nodes())
    rng = random.Random(seed)
    max_size = len(nodes) // 2
    size = rng.randint(2, max_size)
    S = set(rng.sample(nodes, size))
    return S

# Create a ring of cliques of any desired size.
def create_ring_of_cliques(p_num_cliques: int, p_clique_size: int):
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
        for u, v in itertools.combinations(nodes, 2):  # gets all pairs of nodes
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

# Flip operation according to Giakkoupis paper.
def flip_operation(G):
    G = copy.deepcopy(G)
    # 1. Choose an (ordered) pair of adjacent vertices a,b in V (this is the hub-edge)
    a, b = random.choice(list(G.edges))
    # 2. Choose a vertex a' in T(a) (possibly, a' = b)
    a_prime = random.choice(list(set(G.neighbors(a))))
    # 3. If the following two conditions hold: a' in T(a) \ T+(b) AND T(b) \ T+(a) not empty 
    if (a_prime in list(set(G.neighbors(a)) - {b} - set(G.neighbors(b)))) and not (list(set(G.neighbors(b)) - {a} - set(G.neighbors(a))) == []):
        # 3.1. Choose a vertex b' in T(b) \ T+(a)
        b_prime = random.choice(list(set(G.neighbors(b)) - {a} - set(G.neighbors(a))))
        # 3.2. Replace edges (a, a_prime), (b, b_prime) with (a, b_prime), (b, a_prime)
        G.add_edge(a, b_prime)
        G.add_edge(b, a_prime)
        G.remove_edge(a, a_prime)
        G.remove_edge(b, b_prime)
        return G, {(a, a_prime), (b, b_prime)}, {(a, b_prime), (b, a_prime)}
    else:
        return G, None, None

# Flip operation according to Schindelhauer/Mahlmann paper.
def flip_operation_old(G):
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



