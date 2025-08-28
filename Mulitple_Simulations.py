import copy
import math
import numpy as np
import matplotlib.pyplot as plt

import Graph
import Calc_updated


# Auswahl Graph durch User
print("Choose: 1 = Ring of Cliques ; 2 = Random Graph ; 3 = Ring ; 4 = Torus")
index_of_graph = int(input()) - 1
list_of_graphs = ["Ring of Cliques", "Random Graph", "Ring", "Torus"]
type_of_graph = list_of_graphs[index_of_graph]

# Einstellungen
num_simulations = 30
max_flips = 10000  # Optional: Begrenzung für Simulationen
expansion_threshold = 0.5  # Abbruch, sobald spektrale Lücke diesen Wert erreicht

# Graph initialisieren
if type_of_graph == "Ring of Cliques":
    G, pos = Graph.create_ring_of_cliques(5, 5)
    d = 4
elif type_of_graph == "Random Graph":
    G, pos = Graph.create_random_d_regular_graph()
    d = Calc_updated.calculate_d(G)
elif type_of_graph == "Ring":
    G, pos, n = Graph.create_random_even_cycle_graph(seed=42)
    d = 2
else:
    raise ValueError("Ungültiger Graph-Typ")

n = len(G.nodes)
if d > (math.log2(n) ** 2):
    raise ValueError("d ist zu groß für die Knotenzahl")

real_upper_bound = int(n * d * (math.log2(n)) ** 2)
upper_bound = min(real_upper_bound, max_flips)

all_expansions = []
expander_steps = []

# Mehrfache Simulationen
for sim in range(num_simulations):
    current_G = copy.deepcopy(G)
    expansions = []
    expansion = Calc_updated.spectral_expansion(current_G)
    expansions.append(expansion)
    flips_done = 0

    while expansion < expansion_threshold and flips_done < upper_bound:
        new_G, removed, added = Graph.flip_operation(current_G)
        if removed is None or added is None:
            continue
        current_G = new_G
        flips_done += 1
        expansion = Calc_updated.spectral_expansion(current_G)
        expansions.append(expansion)

    all_expansions.append(expansions)
    expander_steps.append(flips_done if expansion >=
                          expansion_threshold else np.nan)

# Daten für den Plot vorbereiten
max_len = max(len(exp) for exp in all_expansions)
padded_expansions = np.array(
    [exp + [np.nan] * (max_len - len(exp)) for exp in all_expansions]
)
mean_expansion = np.nanmean(padded_expansions, axis=0)
std_expansion = np.nanstd(padded_expansions, axis=0)
steps = np.arange(max_len)

best_step = int(np.nanmin(expander_steps))
avg_step = float(np.nanmean(expander_steps))
worst_step = int(np.nanmax(expander_steps))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, mean_expansion, color="black", label="Mean Expansion")
plt.fill_between(
    steps,
    mean_expansion - std_expansion,
    mean_expansion + std_expansion,
    color="skyblue",
    alpha=0.5,
    label="Range of expansion",
)
plt.axvline(best_step, color="green", linestyle="--", label="Best case")
plt.axvline(avg_step, color="black", linestyle="--", label="Average case")
plt.axvline(worst_step, color="red", linestyle="--", label="Worst case")
plt.xlabel("Number of Flip-Operations")
plt.ylabel("Expansion")
plt.title(
    f"Expansion über Flips – Graph: {type_of_graph}, d: {d}, Nodes: {n}, Max Flips: {upper_bound}"
)
plt.legend()
plt.grid(True)
plt.show()
