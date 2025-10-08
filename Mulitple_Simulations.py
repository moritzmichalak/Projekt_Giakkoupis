import networkx as nx
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
from matplotlib.widgets import Button
import Graph
import Calc_updated


# Gtest = nx.random_regular_graph(4, 200, seed=0)
# print("test spec =", Calc_updated.spectral_gap_normalized_sparse(Gtest, 4))

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
print("Draw graphs? y or n")
draw = input()
if draw == "y":
    draw_graphs = True
else:
    draw_graphs = False
print("Run all flips? y or n")
flips = input()
if flips == "y":
    all_flips = True
else:
    all_flips = False
G, pos = Graph.create_ring_of_cliques(number_of_cliques, size_of_cliques)

SAMPLING_RATE_HIGH = math.ceil(n/20)      # alle 10 Flips Spektrum messen
SAMPLING_RATE_LOW = math.ceil(n/10)


num_simulations = 10
max_flips = 5000000000000
criterion = "normalized"

# |spec - epsilon| <= Band gilt als Treffer
# so viele Messungen in Folge im Band
REQUIRED_HITS = 15
COUNT_CONSECUTIVE = True          # True für aufeinanderfolgende Treffer

global_stop_at_flips = None       #

real_upper_bound = int(n * d * (math.log2(n)) ** 2)
upper_bound = min(int(n * d * (math.log2(n)) ** 2), max_flips)
epsilon = Calc_updated.recommend_threshold_by_sampling(n, d)
print(epsilon)
print(upper_bound)


def compute_band(epsilon, rel=0.10, abs_min=1e-3, abs_max=None):
    band = max(abs_min, rel * abs(float(epsilon)))
    if abs_max is not None:
        band = min(band, abs_max)
    return band


# 5 Prozent von epsilon, mindestens 1e-3
THRESHOLD_BAND = compute_band(epsilon, rel=0.05, abs_min=1e-3)

# Arrays für Simulationsergebnisse
all_specs = []
simulations = []  # speichert Daten jeder einzelnen Simulation

# Mehrfache Simulationen
for sim in range(num_simulations):
    nodes = list(G.nodes)
    high_prescision = True
    print("Simulation", sim + 1, "of", num_simulations)
    current_G = G.copy()
    specvals = []
    flip_info = [] if draw_graphs else None
    graphs = [copy.deepcopy(current_G)]

    spec = Calc_updated.spectral_gap_normalized_sparse(current_G, d)
    specvals.append(spec)

    # Zähler für Treffer im Band
    hits_in_band = 0

    flips_done = 0

    # Obergrenze für diese Simulation
    this_upper = upper_bound if global_stop_at_flips is None else global_stop_at_flips

    while flips_done < this_upper:
        current_G, removed, added = Graph.flip_operation(
            current_G, number_of_cliques, size_of_cliques)
        flips_done += 1
        if draw_graphs:
            flip_info.append((removed, added))

        # nur an den Samplingpunkten messen
        measured_now = False
        if high_prescision:
            if flips_done % SAMPLING_RATE_HIGH == 0:
                spec = Calc_updated.spectral_gap_normalized_sparse(
                    current_G, d, tol=5e-3, maxiter=1000)
                measured_now = True
            if spec >= epsilon:
                high_prescision = False
        else:
            if flips_done % SAMPLING_RATE_LOW == 0:
                spec = Calc_updated.spectral_gap_normalized_sparse(
                    current_G, d, tol=2e-2, maxiter=300)
                measured_now = True

        if measured_now:
            in_band = abs(float(spec) - float(epsilon)) <= THRESHOLD_BAND
            if COUNT_CONSECUTIVE:
                hits_in_band = hits_in_band + 1 if in_band else 0
            else:
                if in_band:
                    hits_in_band += 1

            if global_stop_at_flips is None and hits_in_band >= REQUIRED_HITS:
                global_stop_at_flips = flips_done + \
                    max(math.ceil(upper_bound / 20), 50000)
                print(
                    f"Früher Stopp nach {global_stop_at_flips} Flips festgelegt in Simulation {sim + 1}")

                # nur hier trimmen, weil der Wert jetzt sicher nicht None ist
                _target_spec_len = 1 + global_stop_at_flips
                _target_graphs_len = 1 + global_stop_at_flips
                _target_flips_len = global_stop_at_flips

                for s in simulations:
                    if s.get("specvals") is not None and len(s["specvals"]) > _target_spec_len:
                        del s["specvals"][_target_spec_len:]
                    if draw_graphs and s.get("graphs") is not None and len(s["graphs"]) > _target_graphs_len:
                        del s["graphs"][_target_graphs_len:]
                    if draw_graphs and s.get("flip_info") is not None and len(s["flip_info"]) > _target_flips_len:
                        del s["flip_info"][_target_flips_len:]

                for i in range(len(all_specs)):
                    if len(all_specs[i]) > _target_spec_len:
                        del all_specs[i][_target_spec_len:]

                # aktuelle Obergrenze für diese Simulation heruntersetzen
                this_upper = min(this_upper, global_stop_at_flips)

        # Werte anhängen wie gehabt
        specvals.append(float(spec) if np.isfinite(spec) else np.nan)
        if draw_graphs:
            graphs.append(current_G.copy())

        # mindestens alle n Flips, bei großen n noch seltener

        PROGRESS_EVERY = 20000
        if flips_done % PROGRESS_EVERY == 0:
            pct = (sim * this_upper + flips_done) / \
                (num_simulations * this_upper) * 100
            print(f"{pct:.2f} Prozent, Wert: ",
                  spec, ", Threshhold: ", epsilon, ", hits: ", hits_in_band)

    if global_stop_at_flips is not None:
        _target_spec_len = 1 + global_stop_at_flips
        specvals = specvals[:_target_spec_len]
    if draw_graphs and graphs is not None:
        graphs = graphs[:_target_spec_len]
    if draw_graphs and flip_info is not None:
        flip_info = flip_info[:global_stop_at_flips]

    simulations.append({
        "graphs": graphs if draw_graphs else None,
        "specvals": specvals,
        "flip_info": flip_info if draw_graphs else None,
    })
    all_specs.append(specvals)


# In Arrays konvertieren
# all_specs: Liste von Listen
max_len = max(len(s) for s in all_specs) if all_specs else 0
spec_np = np.full((len(all_specs), max_len), np.nan, dtype=float)
for i, s in enumerate(all_specs):
    spec_np[i, :len(s)] = [v if np.isfinite(v) else np.nan for v in s]

# Durchschnittswerte berechnen
mean_spec = np.nanmean(spec_np, axis=0)
mins = np.nanmin(spec_np, axis=0)
maxs = np.nanmax(spec_np, axis=0)

steps = np.arange(spec_np.shape[1])

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

        nx.draw(G, pos, ax=ax_graph, with_labels=True,
                edge_color=edge_colors,
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
    ax.plot(steps, mean_spec, label="Avg Spectral")
    ax.fill_between(steps, mins, maxs, alpha=0.3, label="Range")
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
