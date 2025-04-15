import networkx as xn  # oder: import networkx as xn
import numpy as np
import matplotlib.pyplot as plt
import random

'''
def compute_algebraic_connectivity(G):
    import networkx as nx  # temporär für Matritzenfunktion
    laplacian = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues.sort()
    return eigenvalues[1]  # λ₂
'''
def cut_metrics(G, S):
    # iter(G.nodes) erzeugt einen Iterator über die Knoten des Graphen ; next(iter(...)) holt sich ersten Knoten aus diesem Iterator ; G.degree[node] gibt Grad (also die Anzahl der Nachbarn/Kanten) dieses Knotens zurück.
    # d = G.degree[next(iter(G.nodes))]
    V = set(G.nodes)
    cut_edges = set()
    for u in S:
        for v in G.neighbors(u):
            if v not in S:
                cut_edges.add((u, v))
    cut_size = len(cut_edges)
    min_side = min(len(S), len(V - S))
    # conductance = cut_size / (d * min_side) if min_side > 0 else 0

    # Berechne Cut Strain:
    strain = 0
    for u in G.nodes:
        neighbors = set(G.neighbors(u))
        if not neighbors:
            continue
        # Wieviele der Nachbarknoten liegen in Teilmenge S ?
        alpha_u = len(neighbors.intersection(S)) / len(neighbors)
        strain += alpha_u * (1 - alpha_u)

    # return strain, conductance, cut_edges
    return strain, cut_edges

'''
def expected_cut_size(G, S, u, v):
    """
    Computes the expected change in cut size c_G(S) after a flip at edge (u,v), 
    according to Lemma 16 from the paper.
    """
    if not G.has_edge(u, v):
        return 0

    def beta(ux, vx):
        neighbors = set(G.neighbors(ux))
        if not neighbors:
            return 0
        inter = neighbors.intersection(S)
        inter_common = inter.intersection(G.neighbors(vx))
        alpha_ux = len(inter) / len(neighbors)
        alpha_ux_vx = len(inter_common) / len(neighbors)
        indicator = 1 if vx in S else 0
        return alpha_ux - alpha_ux_vx - indicator / len(neighbors)

    beta_uv = beta(u, v)
    beta_vu = beta(v, u)

    return 2 * (beta_uv - beta_vu)
'''

def expected_cut_strain(G, S):
    """
    Computes the expected strain E[σ_{G'}] after a flip,
    based on Lemma 17 from the paper.
    """
    from itertools import combinations

    V = set(G.nodes)
    d = next(iter(dict(G.degree()).values()))
    m = G.number_of_edges()
    delta = 1 / d

    def beta(u, v):
        neighbors = set(G.neighbors(u))
        if not neighbors:
            return 0
        inter = neighbors.intersection(S)
        inter_common = inter.intersection(G.neighbors(v))
        alpha_u = len(inter) / len(neighbors)
        alpha_uv = len(inter_common) / len(neighbors)
        indicator = 1 if v in S else 0
        return alpha_u - alpha_uv - indicator / len(neighbors)

    def beta_bar(u, v):
        neighbors = set(G.neighbors(u))
        if not neighbors:
            return 0
        inter = neighbors.intersection(V - S)
        inter_common = inter.intersection(G.neighbors(v))
        alpha_u = len(inter) / len(neighbors)
        alpha_uv = len(inter_common) / len(neighbors)
        indicator = 1 if v in V - S else 0
        return alpha_u - alpha_uv - indicator / len(neighbors)

    def rho(u, v):
        neighbors = set(G.neighbors(u))
        common = set(G.neighbors(u)).union({u}).intersection(set(G.neighbors(v)).union({v}))
        return (len(neighbors) - len(neighbors.intersection(common))) / len(neighbors)

    def gamma(u, v):
        if set(G.neighbors(u)).union({u}) == set(G.neighbors(v)).union({v}):
            return 0

        if u in S and v in S:
            return beta_bar(u, v) + beta_bar(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)

        if u in V - S and v in V - S:
            return beta(u, v) + beta(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)

        if (u in S and v in V - S) or (u in V - S and v in S):
            return beta(u, v) + beta_bar(v, u) + 4 * beta(u, v) * beta_bar(v, u) / rho(u, v)

        return 0
<<<<<<< HEAD

=======
    # Funktion aufrufen:
    strain, cut_edges = cut_metrics(G, S)
    '''
>>>>>>> 660f590 (initial commit)
    # Current cut strain
    strain = 0
    for u in G.nodes:
        neighbors = set(G.neighbors(u))
        if not neighbors:
            continue
        alpha_u = len(neighbors.intersection(S)) / len(neighbors)
        strain += alpha_u * (1 - alpha_u)
<<<<<<< HEAD

=======
    '''
>>>>>>> 660f590 (initial commit)
    # Expected change
    term1 = 0
    term2 = 0
    for u, v in G.edges():
        diff = beta(u, v) - beta(v, u)
        term1 += diff ** 2
        term2 += gamma(u, v)

    expected = strain + (4 * delta / m) * term1 - (delta ** 2 / m) * term2
    return expected


def expected_cut_strain_exact(G, S):
    """
    Computes the expected strain E[σ_{G'}] after a flip, using the precise shortcut notation and definitions.
    """
    V = set(G.nodes)
    n = len(V)
    m = G.number_of_edges()
    d = next(iter(dict(G.degree()).values()))  # assumes d-regular
    delta = 1 / d

    def Gamma(u):
        return set(G.neighbors(u))

    def Gamma_plus(u):
        return Gamma(u).union({u})

    def alpha(u):
        neighbors = Gamma(u)
        return len(neighbors.intersection(S)) / len(neighbors) if neighbors else 0

    def alpha_uv(u, v):
        neighbors = Gamma(u)
        return len(neighbors.intersection(S).intersection(Gamma(v))) / len(neighbors) if neighbors else 0

    def alpha_bar_uv(u, v):
        neighbors = Gamma(u)
        return len(neighbors.intersection(V - S).intersection(Gamma(v))) / len(neighbors) if neighbors else 0

    def beta(u, v):
        neighbors = Gamma(u)
        if not neighbors:
            return 0
        intersect = neighbors.intersection(S)
        restricted = intersect - Gamma_plus(v)
        return len(restricted) / len(neighbors)

    def beta_bar(u, v):
        neighbors = Gamma(u)
        if not neighbors:
            return 0
        intersect = neighbors.intersection(V - S)
        restricted = intersect - Gamma_plus(v)
        return len(restricted) / len(neighbors)

    def rho(u, v):
        neighbors = Gamma(u)
        if not neighbors:
            return 0
        return len(neighbors - Gamma_plus(v)) / len(neighbors)

    def gamma(u, v):
        if Gamma_plus(u) == Gamma_plus(v):
            return 0

        if u in S and v in S:
            return beta_bar(u, v) + beta_bar(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)

        if u in V - S and v in V - S:
            return beta(u, v) + beta(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)

        if (u in S and v in V - S) or (u in V - S and v in S):
            return beta(u, v) + beta_bar(v, u) + 4 * beta(u, v) * beta_bar(v, u) / rho(u, v)

        return 0

    # Original strain
    sigma_G = sum(alpha(u) * (1 - alpha(u)) for u in G.nodes)

    # Summation terms
    sum_diff_squared = 0
    sum_gamma = 0
    for u, v in G.edges():
        diff = beta(u, v) - beta(v, u)
        sum_diff_squared += diff ** 2
        sum_gamma += gamma(u, v)

    expected = sigma_G + (4 * delta / m) * sum_diff_squared - (delta ** 2 / m) * sum_gamma
    return expected
