import networkx as nx
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import Graph
import Calc_updated

# Auswahl Graph durch User
print("Choose: 1 = Ring of Cliques ; 2 = Random Graph ; 3 = Ring ; 4 = Torus")
index_of_graph = int(input()) - 1
list_of_graphs = ["Ring of Cliques", "Random Graph", "Ring", "Torus"]
type_of_graph = list_of_graphs[index_of_graph]
# Einstellungen
#type_of_graph = "ring_of_cliques"  # {ring_of_cliques, random, ring}
num_simulations = 30
max_flips = 1000  # Optional: Begrenzung für Simulationen (zum Testen)

# Graph initialisieren
if type_of_graph == "Ring of Cliques":
    G, pos = Graph.create_ring_of_cliques(5, 5)
    cut_set = Graph.generate_random_cut(G)
    # cut_set = {f"C1_{i}" for i in range(5)}
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
if d > (math.log2(n)**2):
    raise ValueError("d ist zu groß für die Knotenzahl")

upper_bound = min(int(n * d * (math.log2(n))**2), max_flips)

# Arrays für Simulationsergebnisse
all_cut_strains = []
all_expected_cut_strains = []
all_cut_sizes = []
all_conductances = []

# Mehrfache Simulationen
for sim in range(num_simulations):
    current_G = copy.deepcopy(G)
    cut_strains = []
    expected_cut_strains = []
    cut_sizes = []
    conductances = []

    strain, conductance, cut_edges = Calc_updated.cut_metrics(
        current_G, cut_set, d)
    expected_strain = Calc_updated.expected_cut_strain_exact(
        current_G, cut_set, d)
    cut_strains.append(strain)
    expected_cut_strains.append(expected_strain)
    cut_sizes.append(len(cut_edges))
    conductances.append(conductance)

    flips_done = 0
    while flips_done < upper_bound:
        new_G, removed, added = Graph.flip_operation(current_G)
        if removed is None or added is None:
            continue
        current_G = new_G
        strain, conductance, cut_edges = Calc_updated.cut_metrics(
            current_G, cut_set, d)
        expected_strain = Calc_updated.expected_cut_strain_exact(
            current_G, cut_set, d)
        cut_strains.append(strain)
        expected_cut_strains.append(expected_strain)
        cut_sizes.append(len(cut_edges))
        conductances.append(conductance)
        flips_done += 1

    all_cut_strains.append(cut_strains)
    all_expected_cut_strains.append(expected_cut_strains)
    all_cut_sizes.append(cut_sizes)
    all_conductances.append(conductances)

# In Arrays konvertieren
cut_strains_np = np.array(all_cut_strains)
expected_strains_np = np.array(all_expected_cut_strains)
cut_sizes_np = np.array(all_cut_sizes)
conductances_np = np.array(all_conductances)

# Durchschnittswerte berechnen
mean_cut_strain = np.mean(cut_strains_np, axis=0)
std_cut_strain = np.std(cut_strains_np, axis=0)

mean_expected_strain = np.mean(expected_strains_np, axis=0)
std_expected_strain = np.std(expected_strains_np, axis=0)

mean_cut_size = np.mean(cut_sizes_np, axis=0)
std_cut_size = np.std(cut_sizes_np, axis=0)

mean_conductance = np.mean(conductances_np, axis=0)
std_conductance = np.std(conductances_np, axis=0)

# Anfangsgraph mit Cut anzeigen
plt.figure(figsize=(8, 6))
node_colors = ["yellow" if n in cut_set else "lightblue" for n in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        node_size=500, font_size=10)
plt.title("Anfangsgraph mit Cut")
plt.show()

# Plots anzeigen
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
steps = range(len(mean_cut_strain))

axs[0].plot(steps, mean_cut_size, label="Avg Cut Size", color="orange")
axs[0].fill_between(steps, mean_cut_size - std_cut_size,
                    mean_cut_size + std_cut_size, color="orange", alpha=0.3)

axs[0].legend()
axs[0].set_title("Durchschnittliche Cut Size")
axs[0].grid(True)

axs[1].plot(steps, mean_cut_strain, label="Avg Cut Strain", color="blue")
axs[1].fill_between(steps, mean_cut_strain - std_cut_strain,
                    mean_cut_strain + std_cut_strain, color="blue", alpha=0.3)
axs[1].legend()
axs[1].set_title("Durchschnittliche Cut Strain")
axs[1].grid(True)

axs[2].plot(steps, mean_expected_strain,
            label="Avg Expected Strain", color="purple")
axs[2].fill_between(steps, mean_expected_strain - std_expected_strain,
                    mean_expected_strain + std_expected_strain, color="purple", alpha=0.3)
axs[2].legend()
axs[2].set_title("Durchschnittliche Erwartete Cut Strain")
axs[2].grid(True)

axs[3].plot(steps, mean_conductance, label="Avg Conductance", color="green")
axs[3].fill_between(steps, mean_conductance - std_conductance,
                    mean_conductance + std_conductance, color="green", alpha=0.3)
axs[3].legend()
axs[3].set_title("Conductance")
axs[3].grid(True)

plt.xlabel("Flip-Schritte")
plt.tight_layout()
plt.show()


