import networkx as xn
import numpy as np
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.sparse import identity


def cut_metrics(G, S, d):
    V = set(G.nodes)
    cut_edges = set()
    for u in S:
        for v in G.neighbors(u):
            if v not in S:
                cut_edges.add((u, v))
    cut_size = len(cut_edges)
    min_subset = min(len(S), len(V - S))
    # conductance = cut_size / (d * min_subset) if min_subset > 0 else 0
    conductance = 0

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

# Compute all possible regular degrees d > 2 for a graph with n nodes.

# def calculate_expected_cut_size(n):


def calculate_possible_d(n):
    possible_d = []
    for i in range(3, n):
        ganzzahlig = (n/(i+1)) % 1
        if ganzzahlig == 0:
            possible_d.append(i)
    return possible_d


def calculate_expected_cut_strain_alternative(G, S, d, strain):
    V = set(G.nodes)
    n = len(V)
    m = G.number_of_edges()
    neighbors = {u: set(G.neighbors(u)) for u in G.nodes()}
    # expected_cut_strain_alternative = 0
    # cut_strain_scenario = strain
    strains_a_prime = 0
    # sum_counter = 0
    counter = 0
    for a, b in G.edges:  # number of iterations: m
        Na = neighbors[a]
        Nb = neighbors[b]
        for a_prime in Na:  # number of iterations: d
            sum_a_prime = 0
            if (a_prime in (Na - {b} - Nb)) and (len(Nb - {a} - Na) != 0):
                averaged_strain_a_prime = 0
                for b_prime in (Nb - {a} - Na):
                    strain_b_prime = strain
                    Na_prime = neighbors[a_prime]
                    Nb_prime = neighbors[b_prime]
                    # The cut runs between a and b OR a' and b' => flip-operation changes cut strain.
                    if ((a in S) ^ (b in S)) or ((a_prime in S) ^ (b_prime in S)):
                        # Calculate old values:
                        alpha_a = len(Na.intersection(S)) / len(Na)
                        alpha_b = len(Nb.intersection(S)) / len(Nb)
                        alpha_a_prime = len(
                            Na_prime.intersection(S)) / len(Na_prime)
                        alpha_b_prime = len(
                            Nb_prime.intersection(S)) / len(Nb_prime)
                        # Substract old values
                        strain_b_prime -= (alpha_a * (1 - alpha_a) + alpha_b * (1 - alpha_b) + alpha_a_prime * (
                            1 - alpha_a_prime) + alpha_b_prime * (1 - alpha_b_prime))
                        # Calculate new values:
                        if (a in S) and (b not in S) and (a_prime in S) and (b_prime not in S):  # I
                            alpha_a = (len(Na.intersection(S)) - 1) / len(Na)
                            alpha_b = (len(Nb.intersection(S)) + 1) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S)) - 1) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S)) + 1) / len(Nb_prime)
                        elif (a in S) and (b not in S) and (a_prime not in S) and (b_prime in S):  # II
                            alpha_a = (len(Na.intersection(S)) + 1) / len(Na)
                            alpha_b = (len(Nb.intersection(S)) - 1) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S)) - 1) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S)) + 1) / len(Nb_prime)
                        elif (a not in S) and (b in S) and (a_prime in S) and (b_prime not in S):  # III
                            alpha_a = (len(Na.intersection(S)) - 1) / len(Na)
                            alpha_b = (len(Nb.intersection(S)) + 1) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S)) + 1) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S)) - 1) / len(Nb_prime)
                        elif (a not in S) and (b in S) and (a_prime not in S) and (b_prime in S):  # IV
                            alpha_a = (len(Na.intersection(S)) + 1) / len(Na)
                            alpha_b = (len(Nb.intersection(S)) - 1) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S)) + 1) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S)) - 1) / len(Nb_prime)
                        # Hier kann man noch weiter zusammenfassen:
                        elif ((a not in S) and (b in S) and (a_prime not in S) and (b_prime not in S)) or ((a not in S) and (b in S) and (a_prime in S) and (b_prime in S)):  # V, XII
                            alpha_a = (len(Na.intersection(S))) / len(Na)
                            alpha_b = (len(Nb.intersection(S))) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S)) + 1) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S)) - 1) / len(Nb_prime)
                        elif ((a not in S) and (b not in S) and (a_prime in S) and (b_prime not in S)) or ((a in S) and (b in S) and (a_prime in S) and (b_prime not in S)):  # VI, VIII
                            alpha_a = (len(Na.intersection(S)) - 1) / len(Na)
                            alpha_b = (len(Nb.intersection(S)) + 1) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S))) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S))) / len(Nb_prime)
                        elif ((a not in S) and (b not in S) and (a_prime not in S) and (b_prime in S)) or ((a in S) and (b in S) and (a_prime not in S) and (b_prime in S)):  # VII, IX
                            alpha_a = (len(Na.intersection(S)) + 1) / len(Na)
                            alpha_b = (len(Nb.intersection(S)) - 1) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S))) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S))) / len(Nb_prime)
                        elif ((a in S) and (b not in S) and (a_prime in S) and (b_prime in S)) or ((a in S) and (b not in S) and (a_prime not in S) and (b_prime not in S)):  # X, XI
                            alpha_a = (len(Na.intersection(S))) / len(Na)
                            alpha_b = (len(Nb.intersection(S))) / len(Nb)
                            alpha_a_prime = (
                                len(Na_prime.intersection(S))-1) / len(Na_prime)
                            alpha_b_prime = (
                                len(Nb_prime.intersection(S)) + 1) / len(Nb_prime)
                        else:
                            raise Exception(
                                "Problem with function alternative exp. cut strain")
                        # add new values
                        # print("Cut strain: ", strain,"Neu berechneter Cut strain: ", cut_strain_scenario)
                        strain_b_prime += (alpha_a * (1 - alpha_a) + alpha_b * (1 - alpha_b) + alpha_a_prime * (
                            1 - alpha_a_prime) + alpha_b_prime * (1 - alpha_b_prime))
                        sum_a_prime += strain_b_prime
                    else:
                        sum_a_prime += strain_b_prime
                averaged_strain_a_prime = sum_a_prime / len(Nb - {a} - Na)
                strains_a_prime += averaged_strain_a_prime
            else:
                strains_a_prime += strain
    return strains_a_prime / (m*d)
    # For symmetry:


def expected_cut_strain_exact(G, S, d, strain):
    V = set(G.nodes)
    n = len(V)
    m = G.number_of_edges()
    # d = next(iter(dict(G.degree()).values()))  # assumes d-regular
    delta = 1 / d

    neighbors = {u: set(G.neighbors(u)) for u in G.nodes()}

    def Gamma(u):
        return neighbors[u]

    def Gamma_plus(u):
        return neighbors[u] | {u}

    def alpha(u):
        Nu = neighbors[u]
        return len(Nu & S) / len(Nu) if Nu else 0

    def alpha_uv(u, v):
        Nu = neighbors[u]
        return len(Nu & S & neighbors[v]) / len(Nu) if Nu else 0

    def alpha_bar_uv(u, v):
        Nu = neighbors[u]
        return len(Nu & (V - S) & neighbors[v]) / len(Nu) if Nu else 0

    def beta(u, v):
        Nu = neighbors[u]
        if not Nu:
            return 0
        restricted = (Nu & S) - Gamma_plus(v)
        return len(restricted) / len(Nu)

    def beta_bar(u, v):
        Nu = neighbors[u]
        if not Nu:
            return 0
        restricted = (Nu & (V - S)) - Gamma_plus(v)
        return len(restricted) / len(Nu)

    def rho(u, v):
        Nu = neighbors[u]
        if not Nu:
            return 0
        return len(Nu - Gamma_plus(v)) / len(Nu)

    def gamma(u, v):
        if Gamma_plus(u) == Gamma_plus(v):
            return 0
        if (u in S) and (v in S):
            return beta_bar(u, v) + beta_bar(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)
        if (u not in S) and (v not in S):
            return beta(u, v) + beta(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)
        # über der Schnittkante:
        return beta(u, v) + beta_bar(v, u) + 4 * beta(u, v) * beta_bar(v, u) / rho(u, v)
    '''
        if Gamma_plus(u) == Gamma_plus(v):
            return 0

        if u in S and v in S:
            return beta_bar(u, v) + beta_bar(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)

        if u in V - S and v in V - S:
            return beta(u, v) + beta(v, u) + 2 * (beta(u, v) * beta_bar(v, u) + beta_bar(u, v) * beta(v, u)) / rho(u, v)

        if (u in S and v in V - S) or (u in V - S and v in S):
            return beta(u, v) + beta_bar(v, u) + 4 * beta(u, v) * beta_bar(v, u) / rho(u, v)

        return 0
    '''
    # Tatsächlicher cut-strain:

    # cut_strain, conductance, cut_edges = cut_metrics(G, S, d)
    # sigma_G = sum(alpha(u) * (1 - alpha(u)) for u in G.nodes)

    # Summation terms
    sum_diff_squared = 0
    sum_gamma = 0
    for u, v in G.edges():
        diff = beta(u, v) - beta(v, u)
        sum_diff_squared += diff * diff
        sum_gamma += gamma(u, v)

    return strain + (4 * delta / m) * sum_diff_squared - (delta ** 2 / m) * sum_gamma


def spectral_gap_normalized_sparse(G, d=None, tol=1e-3, maxiter=1000, v0=None):
    """
     Normalized spectral gap gamma, equal to 1.0 minus lambda two of A divided by d.    
     A is the CSR adjacency matrix, d is the degree, if None, d is estimated from the degree mode.Returns float, 
     np.nan if there are problems.
    """
    d = float(d)
    if not np.isfinite(d) or d <= 0:
        return np.nan
    A = xn.to_scipy_sparse_array(G, dtype=float, format="csr")
    try:
        vals = eigsh(A, k=2, which="LA", return_eigenvectors=False,
                     tol=tol, maxiter=maxiter, v0=v0)
    except ArpackNoConvergence as e:
        vals = e.eigenvalues
        if vals is None or len(vals) < 2:
            vals, _ = eigsh(A, k=2, which="LA", return_eigenvectors=True, tol=max(
                tol, 5e-3), maxiter=maxiter*2)
    vals = np.sort(np.asarray(vals, dtype=float))
    if vals.size < 2 or not np.all(np.isfinite(vals)):
        return np.nan
    lam2 = float(vals[-2])
    gamma = 1.0 - lam2 / d
    return float(gamma) if np.isfinite(gamma) else np.nan


def _one_trial(n, d, seed):
    G = xn.random_regular_graph(d, n, seed=int(seed))
    return spectral_gap_normalized_sparse(G, d)


# Calculates threshold by genereating 10 random graphs and calculating there normalized spectral gap. Optimized by threading. Quantile can be choosen.
def recommend_threshold_by_sampling(n, d, trials=10, quantile=0.60, seed=42, n_jobs=-1):
    if n < 3:
        raise ValueError(
            "n muss mindestens 3 sein, damit k=2 bei eigsh funktioniert")
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000, size=trials)
    gammas = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_trial)(n, d, s) for s in seeds
    )
    gammas = np.asarray(gammas, dtype=float)
    thr = float(np.quantile(gammas, quantile))
    stats = {
        "median": float(np.median(gammas)),
        "p25": float(np.quantile(gammas, 0.25)),
        "p75": float(np.quantile(gammas, 0.75)),
        "mean": float(np.mean(gammas)),
        "std": float(np.std(gammas)),
        "samples": int(trials),
    }
    return thr
