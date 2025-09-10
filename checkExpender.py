import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh


class ExpanderChecker:
    """Check spectral expansion properties of an undirected graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph (undirected).
    threshold : float, optional
        Expansion threshold :math:`\varepsilon` for expander detection.
    tol : float, optional
        Tolerance passed to :func:`scipy.sparse.linalg.eigsh`.
    maxiter : int, optional
        Maximum iterations for the eigensolver.
    """

    def __init__(self, G: nx.Graph, threshold: float = 0.1, tol: float = 1e-4, maxiter: int = 300) -> None:
        if G.is_directed():
            raise nx.NetworkXError(
                "ExpanderChecker only supports undirected graphs")
        self.G = G.copy()
        self.threshold = threshold
        self.tol = tol
        self.maxiter = maxiter
        self._fiedler_vec = None
        self._lambda2 = None

    def set_threshold(self, eps: float) -> None:
        """Update the expansion threshold."""
        self.threshold = eps

    def flip_edge(self, u, v, add: bool) -> None:
        """Add or remove an edge (u, v) from the graph."""
        if add:
            self.G.add_edge(u, v)
        else:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
        # Invalidate cached spectral data after modification
        self._fiedler_vec = None
        self._lambda2 = None

    def _normalized_laplacian(self):
        return nx.normalized_laplacian_matrix(self.G)

    def _spectral_sweep_cut(self, vec):
        """Return a sweep cut from a given vector and its conductance."""
        nodes = list(self.G.nodes())
        order = np.argsort(vec)
        sorted_nodes = [nodes[i] for i in order]
        m2 = 2 * self.G.number_of_edges()
        best_cond = np.inf
        best_set = set()
        vol = 0
        S = set()
        for node in sorted_nodes[:-1]:  # ensure both sides non-empty
            S.add(node)
            vol += self.G.degree(node)
            if vol == 0 or vol == m2:
                continue
            boundary = nx.cut_size(self.G, S)
            cond = boundary / min(vol, m2 - vol)
            if cond < best_cond:
                best_cond = cond
                best_set = S.copy()
        return best_set, best_cond

    def check_expansion(self) -> dict:
        """Compute spectral expansion and Cheeger bounds."""
        L = self._normalized_laplacian()
        n = self.G.number_of_nodes()
        if n < 2:
            raise ValueError("Graph must have at least two nodes")
        v0 = self._fiedler_vec if self._fiedler_vec is not None and len(
            self._fiedler_vec) == n else None
        vals, vecs = eigsh(L, k=2, which="SM", tol=self.tol,
                           maxiter=self.maxiter, v0=v0)
        # Eigenvalues are returned in ascending order
        lambda2 = float(vals[1])
        fiedler = vecs[:, 1]
        self._lambda2 = lambda2
        self._fiedler_vec = fiedler
        lower = lambda2 / 2.0
        upper = float(np.sqrt(2.0 * lambda2))
        is_expander = lambda2 >= self.threshold
        cut, conductance = self._spectral_sweep_cut(fiedler)
        return {
            "lambda2": lambda2,
            "lower_bound": lower,
            "upper_bound": upper,
            "is_expander": is_expander,
            "cut": cut,
            "conductance": conductance,
        }

    def get_fiedler_vector(self) -> np.ndarray:
        """Return the Fiedler vector (computes it if necessary)."""
        if self._fiedler_vec is None:
            self.check_expansion()
        return self._fiedler_vec


if __name__ == "__main__":
    # Demonstration
    print("Path graph (poor expander)")
    G_path = nx.path_graph(20)
    checker_path = ExpanderChecker(G_path, threshold=0.5)
    res_path = checker_path.check_expansion()
    print(res_path)

    print("\nRandom regular graph (better expander)")
    G_rr = nx.random_regular_graph(d=3, n=20, seed=42)
    checker_rr = ExpanderChecker(G_rr, threshold=0.5)
    res_rr = checker_rr.check_expansion()
    print(res_rr)
