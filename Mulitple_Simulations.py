import networkx as nx
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
from matplotlib.widgets import Button
import Graph
import Calc_updated

# -------- Spektralmetrik --------


def spectral_value(G, d, criterion="normalized"):
    """
    Liefert den Spektralwert für die Expanderentscheidung.
    criterion gleich "normalized": γ gleich λ zwei von L, L gleich I minus A geteilt durch d
    criterion gleich "adjacency": relative Lücke gleich (d minus λ Stern) geteilt durch d
    """
    A = nx.to_numpy_array(G, dtype=float)
    if criterion == "normalized":
        n = A.shape[0]
        L = np.eye(n) - A / d
        vals = np.linalg.eigvalsh(L)            # aufsteigend, λ eins gleich 0
        return float(vals[1])
    elif criterion == "adjacency":
        vals = np.linalg.eigvalsh(A)            # aufsteigend
        # größter Eigenwert ist d, λ Stern ist maximale Größe der übrigen
        lam_star = np.max(np.abs(vals[:-1])) if len(vals) > 1 else 0.0
        gap_rel = (d - lam_star) / d
        return float(gap_rel)
    else:
        raise ValueError('criterion muss "normalized" oder "adjacency" sein')


"""
# Auswahl Graph durch User
print("Choose: 1 = Ring of Cliques ; 2 = Random Graph ; 3 = Ring")
index_of_graph = int(input()) - 1
list_of_graphs = ["Ring of Cliques", "Random Graph", "Ring"]
type_of_graph = list_of_graphs[index_of_graph]

# Einstellungen
         # alternativ "adjacency"
# Schwellwert für Expander

# Graph initialisieren
if type_of_graph == "Ring of Cliques":
    G, pos = Graph.create_ring_of_cliques(8, 5)
    cut_set = {f"C1_{i}" for i in range(5)}     # nur noch fürs Zeichnen
    d = 4
elif type_of_graph == "Random Graph":
    G, pos = Graph.create_random_d_regular_graph()
    cut_set = Graph.generate_random_cut(G)
    d = Calc_updated.calculate_d(G)
elif type_of_graph == "Ring":
    G, pos, n = Graph.create_random_even_cycle_graph(seed=42)
    d = 2
    cut_set = Graph.generate_random_cut(G)
else:
    raise ValueError("Ungültiger Graph-Typ")

n = len(G.nodes)
if d > (math.log2(n) ** 2):
    raise ValueError("d ist zu groß für die Knotenzahl")

"""

print("Choose number of nodes:")
n = int(input())
possible_d = Calc_updated.calculate_possible_d(n)
print("Choose degree from ", possible_d)
chosen_d = int(input())
for i in possible_d:
    if chosen_d == i:
        d = chosen_d
number_of_cliques = int(n / (d+1))
size_of_cliques = d+1
print("You chose a graph with ", number_of_cliques,
      " cliques, each with a size of", size_of_cliques, " nodes.")
G, pos = Graph.create_ring_of_cliques(number_of_cliques, size_of_cliques)


num_simulations = 20
max_flips = 5000
criterion = "normalized"

real_upper_bound = int(n * d * (math.log2(n)) ** 2)
upper_bound = min(int(n * d * (math.log2(n)) ** 2), max_flips)
epsilon = Calc_updated.recommend_threshold_by_sampling(n, d)
print(epsilon)

# Arrays für Simulationsergebnisse
all_specs = []
simulations = []  # speichert Daten jeder einzelnen Simulation

# Mehrfache Simulationen
for sim in range(num_simulations):
    print("Simulation ", sim+1, " of ", num_simulations)
    current_G = copy.deepcopy(G)
    graphs = [copy.deepcopy(current_G)]
    specvals = []
    flip_info = [(set(), set())]

    spec = Calc_updated.spectral_gap_normalized_sparse(current_G, d)
    specvals.append(spec)

    flips_done = 0
    while flips_done < upper_bound:
        new_G, removed, added = Graph.flip_operation(current_G)
        if removed is None or added is None:
            continue
        current_G = new_G
        graphs.append(copy.deepcopy(current_G))
        spec = Calc_updated.spectral_gap_normalized_sparse(current_G, d)
        specvals.append(spec)
        flip_info.append((removed, added))
        flips_done += 1
        if (flips_done % 1000 == 0):
            print(flips_done, "Flips done")

    simulations.append({
        "graphs": graphs,
        "specvals": specvals,
        "flip_info": flip_info,
    })

    all_specs.append(specvals)

# In Arrays konvertieren
max_len = max(len(c) for c in all_specs)
spec_np = np.full((len(all_specs), max_len), np.nan)
for i, c in enumerate(all_specs):
    spec_np[i, :len(c)] = c

# Durchschnittswerte berechnen
mean_spec = np.nanmean(spec_np, axis=0)
std_spec = np.nanstd(spec_np, axis=0)

# Extremfälle bestimmen, basierend auf erstem Zeitpunkt, an dem der Threshold erreicht wird
threshold = epsilon

# erster Index i mit s[i] größer gleich threshold, sonst np.inf
first_hit = []
for sim in simulations:
    s = sim["specvals"]
    idx = next((i for i, v in enumerate(s) if v >= threshold), np.inf)
    first_hit.append(idx)

# Kandidaten, die den Threshold erreichen
reachable = [i for i, t in enumerate(first_hit) if np.isfinite(t)]

if reachable:
    # Best Case, erreicht am frühesten
    # Tie Breaker, kürzerer Lauf zuerst, danach höhere Endausprägung
    best_index = min(
        reachable,
        key=lambda i: (first_hit[i],
                       len(simulations[i]["specvals"]),
                       -simulations[i]["specvals"][-1])
    )
    # Worst Case, erreicht am spätesten
    # Tie Breaker, längerer Lauf zuerst, danach niedrigere Endausprägung
    worst_index = max(
        reachable,
        key=lambda i: (first_hit[i],
                       -len(simulations[i]["specvals"]),
                       simulations[i]["specvals"][-1])
    )
else:
    # Niemand erreicht den Threshold, Fallback
    # Best Case, größter Endwert
    best_index = int(np.argmax([sim["specvals"][-1] for sim in simulations]))
    # Worst Case, kleinster Endwert
    worst_index = int(np.argmin([sim["specvals"][-1] for sim in simulations]))


# Anzeige einer Simulation


def view_simulation(sim_index, back_callback):
    data = simulations[sim_index]
    graphs = data["graphs"]
    specvals = data["specvals"]
    flip_info = data["flip_info"]

    fig, ax_graph = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)
    current_step = [0]

    def draw_step(index):
        ax_graph.clear()
        G = graphs[index]
        removed_edges, added_edges = flip_info[index]
        edge_colors = []
        for u, v in G.edges():
            edge = tuple(sorted((u, v)))
            if removed_edges and edge in {tuple(sorted(e)) for e in removed_edges}:
                edge_colors.append("blue")
            elif added_edges and edge in {tuple(sorted(e)) for e in added_edges}:
                edge_colors.append("red")
            else:
                edge_colors.append("black")

        node_colors = [
            "yellow" if n in cut_set else "lightblue" for n in G.nodes()]

        nx.draw(G, pos, ax=ax_graph, with_labels=True,
                node_color=node_colors, edge_color=edge_colors,
                node_size=500, font_size=8)

        ax_graph.set_title(
            f"Flip Schritt {index}, Spektralwert {specvals[index]:.3f}\n"
            f"Removed: {removed_edges}, Added: {added_edges}"
        )
        plt.draw()

    def next_step(event):
        current_step[0] = (current_step[0] + 1) % len(graphs)
        draw_step(current_step[0])

    def prev_step(event):
        current_step[0] = (current_step[0] - 1) % len(graphs)
        draw_step(current_step[0])

    axprev = plt.axes([0.2, 0.02, 0.15, 0.05])
    axnext = plt.axes([0.65, 0.02, 0.15, 0.05])
    axback = plt.axes([0.425, 0.02, 0.15, 0.05])
    bprev = Button(axprev, '⟵ Previous')
    bnext = Button(axnext, 'Next ⟶')
    bback = Button(axback, 'Back')
    bprev.on_clicked(prev_step)
    bnext.on_clicked(next_step)

    def go_back(event):
        plt.close(fig)
        back_callback()

    bback.on_clicked(go_back)

    draw_step(current_step[0])
    plt.show()


def show_main_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.2)
    steps = range(len(mean_spec))

    best_case = simulations[best_index]["specvals"]
    worst_case = simulations[worst_index]["specvals"]

    ax.plot(range(len(best_case)), best_case, label="Best Case", color="blue")
    ax.plot(range(len(worst_case)), worst_case,
            label="Worst Case", color="red")
    ax.plot(steps, mean_spec, label="Avg Spectral", color="green")
    ax.fill_between(
        steps,
        mean_spec - std_spec,
        mean_spec + std_spec,
        color="green",
        alpha=0.3,
    )
    ax.axhline(threshold, color="gray", linestyle="--",
               label=f"Threshold {threshold}")
    ax.legend()
    ax.set_title("Spektrale Expansion")
    ax.set_xlabel("Flip Schritte")
    ax.set_ylabel("Spektralwert")
    ax.grid(True)

    fig.suptitle(
        f"Simulationsergebnisse, {num_simulations} Läufe, Graph: Ring of Cliques, d: {d}, Nodes: {n}, Upper Bound: {real_upper_bound}",
        fontsize=14,
        y=0.97,
    )

    ax_best = plt.axes([0.25, 0.02, 0.2, 0.07])
    ax_worst = plt.axes([0.55, 0.02, 0.2, 0.07])
    b_best = Button(ax_best, 'Best Case')
    b_worst = Button(ax_worst, 'Worst Case')

    def open_best(event):
        plt.close(fig)
        view_simulation(best_index, show_main_plot)

    def open_worst(event):
        plt.close(fig)
        view_simulation(worst_index, show_main_plot)

    b_best.on_clicked(open_best)
    b_worst.on_clicked(open_worst)

    plt.show()


show_main_plot()
