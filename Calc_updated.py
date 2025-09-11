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
    # return strain, cut_edges


def calculate_d(G):
    d = next(iter(dict(G.degree()).values()))
    return d

# 28.08.25 / 03.09.25: Erwarteten Cut Strain-Wert durch überprüfen jeder Kante berechnen:
def calculate_expected_cut_strain_alternative(G, S, d, strain):
    V = set(G.nodes)
    n = len(V)
    m = G.number_of_edges()
    expected_cut_strain_alternative = 0
    # cut_strain_scenario = strain
    for a, b in G.edges: # number of iterations = m
        for a_prime in set(G.neighbors(a)): # number of iterations = d
            # Does a' permit flip-operation?
            if (a_prime in list(set(G.neighbors(a)) - {b} - set(G.neighbors(b)))) and not (list(set(G.neighbors(b)) - {a} - set(G.neighbors(a))) == []):
                # a' permits flip-operation.
                sum = 0
                for b_prime in set(G.neighbors(b)) - {a} - set(G.neighbors(a)): # variable number of iterations.
                    cut_strain_scenario = strain
                    # The cut runs between a' und b' -> flip-operation changes cut strain.
                    if (a_prime in S) ^ (b_prime in S):
                        # print("(a,b) = ", a, b, "a' = ", a_prime, "b' = ", b_prime,)
                        # Calculate old values: 
                        alpha_a = len(set(G.neighbors(a)).intersection(S)) / len(set(G.neighbors(a)))
                        alpha_b = len(set(G.neighbors(b)).intersection(S)) / len(set(G.neighbors(b)))
                        alpha_a_prime = len(set(G.neighbors(a_prime)).intersection(S)) / len(set(G.neighbors(a_prime)))
                        alpha_b_prime = len(set(G.neighbors(b_prime)).intersection(S)) / len(set(G.neighbors(b_prime)))
                        # print("alpha-Werte vorher: ", alpha_a, alpha_b, alpha_a_prime, alpha_b_prime)
                        # Substract old values:
                        cut_strain_scenario -= (alpha_a * (1 - alpha_a) + alpha_b * (1 - alpha_b) + alpha_a_prime * (1 - alpha_a_prime) + alpha_b_prime * (1 - alpha_b_prime))
                        # Calculate new values: 
                        if a_prime in S: # (a_prime  in S and b_prime not in S)
                            alpha_a = (len(set(G.neighbors(a)).intersection(S)) - 1) / len(set(G.neighbors(a)))
                            alpha_b = (len(set(G.neighbors(b)).intersection(S)) + 1) / len(set(G.neighbors(b)))
                            alpha_a_prime = (len(set(G.neighbors(a_prime)).intersection(S)) - 1) / len(set(G.neighbors(a_prime)))
                            alpha_b_prime = (len(set(G.neighbors(b_prime)).intersection(S)) + 1) / len(set(G.neighbors(b_prime)))
                        else:  # (a_prime not in S and b_prime  in S)
                            alpha_a = (len(set(G.neighbors(a)).intersection(S)) + 1) / len(set(G.neighbors(a)))
                            alpha_b = (len(set(G.neighbors(b)).intersection(S)) - 1) / len(set(G.neighbors(b)))
                            alpha_a_prime = (len(set(G.neighbors(a_prime)).intersection(S)) + 1) / len(set(G.neighbors(a_prime)))
                            alpha_b_prime = (len(set(G.neighbors(b_prime)).intersection(S)) - 1) / len(set(G.neighbors(b_prime)))
                        #print("alpha-Werte nachher: ", alpha_a, alpha_b, alpha_a_prime, alpha_b_prime)
                        # cut strain for specific 3-node-path : (a',a) - (a,b) - (b,b')
                        cut_strain_scenario += (alpha_a * (1 - alpha_a) + alpha_b * (1 - alpha_b) + alpha_a_prime * (1 - alpha_a_prime) + alpha_b_prime * (1 - alpha_b_prime))
                        #print("Cut strain: ", strain,"Neu berechneter Cut strain: ", cut_strain_scenario)
                    # sum of all cut strain over all possible b' 
                    sum += cut_strain_scenario
                # weighted cut strain
                averaged_new_strain = (sum / len(set(G.neighbors(b)) - {a} - set(G.neighbors(a))))
                # print("averaged_new_strain = ", averaged_new_strain)
                expected_cut_strain_alternative += (averaged_new_strain / (m*d))
            else: 
                # a' does not permit flip-operation.
                expected_cut_strain_alternative += (strain / (m*d))
    # print("expected_cut_strain_alternative: ", expected_cut_strain_alternative)    
    return expected_cut_strain_alternative
                                




def expected_cut_strain_exact(G, S, d, strain):
    V = set(G.nodes)
    n = len(V)
    m = G.number_of_edges()
    # d = next(iter(dict(G.degree()).values()))  # assumes d-regular
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

    # Tatsächlicher cut-strain:

    # cut_strain, conductance, cut_edges = cut_metrics(G, S, d)
    # sigma_G = sum(alpha(u) * (1 - alpha(u)) for u in G.nodes)

    # Summation terms
    sum_diff_squared = 0
    sum_gamma = 0
    for u, v in G.edges():
        diff = beta(u, v) - beta(v, u)
        sum_diff_squared += diff ** 2
        sum_gamma += gamma(u, v)

    expected = strain + (4 * delta / m) * sum_diff_squared - \
        (delta ** 2 / m) * sum_gamma
    return expected


def spectral_expansion(G):
    """Berechnet die spektrale Expansion eines d-regulären Graphen.

    Für einen d-regulären Graphen ist der größte Eigenwert der Adjazenzmatrix
    gleich ``d``. Die Expansion wird als ``d - lambda_2`` definiert, wobei
    ``lambda_2`` der zweitgrößte Eigenwert ist.

    Args:
        G: Ein ungerichteter d-regulärer NetworkX-Graph.

    Returns:
        float: Die spektrale Lücke ``d - lambda_2``.
    """

    A = xn.to_numpy_array(G)
    eigenvalues = np.linalg.eigvalsh(A)
    d = eigenvalues[-1]
    lambda2 = eigenvalues[-2]
    return d - lambda2
