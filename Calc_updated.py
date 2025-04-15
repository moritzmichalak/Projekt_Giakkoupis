import networkx as xn  # oder: import networkx as xn
import numpy as np
import matplotlib.pyplot as plt
import random

def cut_metrics(G, S, d):
    # iter(G.nodes) erzeugt einen Iterator über die Knoten des Graphen ; next(iter(...)) holt sich ersten Knoten aus diesem Iterator ; G.degree[node] gibt Grad (also die Anzahl der Nachbarn/Kanten) dieses Knotens zurück.
    # d = G.degree[next(iter(G.nodes))]
    V = set(G.nodes)
    cut_edges = set()
    for u in S:
        for v in G.neighbors(u):
            if v not in S:
                cut_edges.add((u, v))
    cut_size = len(cut_edges)
    min_subset = min(len(S), len(V - S))
    conductance = cut_size / (d * min_subset) if min_subset > 0 else 0

    # Berechne Cut Strain:
    strain = 0
    for u in G.nodes:
        neighbors = set(G.neighbors(u))
        if not neighbors:
            continue
        # Wieviele der Nachbarknoten liegen in Teilmenge S ?
        alpha_u = len(neighbors.intersection(S)) / len(neighbors)
        strain += alpha_u * (1 - alpha_u)

    return strain, conductance, cut_edges
    #return strain, cut_edges

def calculate_d(G):
    # TO DO: Obergrenze aus Paper für d: O(log^2 n)
    d = next(iter(dict(G.degree()).values()))
    return d

def expected_cut_strain_exact(G, S, d):
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

