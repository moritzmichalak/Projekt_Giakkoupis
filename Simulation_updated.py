import networkx as nx
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import Button
import Graph
import math
import Calc_updated

# Auswahl Graph durch User
print("Choose: 1 = Ring of Cliques ; 2 = Random Graph ; 3 = Ring ; 4 = Torus")
index_of_graph = int(input()) - 1
list_of_graphs = ["Ring of Cliques", "Random Graph", "Ring", "Torus"]
type_of_graph = list_of_graphs[index_of_graph]
# Einstellungen
# type_of_graph = "ring_of_cliques"  # {ring_of_cliques, random, ring}
num_simulations = 30
max_flips = 1000  # Optional: Begrenzung für Simulationen (zum Testen)

# Graph initialisieren
if type_of_graph == "Ring of Cliques":
    G, pos = Graph.create_ring_of_cliques(5, 5)
    # cut_set = Graph.generate_random_cut(G)
    # smallest cut:
    # cut_set = {f"C1_{i}" for i in range(5)}
    # biggest cut:
    cut_set = {"C2_2", "C2_3", "C2_4", "C3_0", "C3_4", "C3_3", "C4_1",
               "C4_0", "C4_4", "C0_2", "C0_1", "C0_0", "C1_3", "C1_2", "C1_1"}

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

n = len(G.nodes)  # Anzahl der Knoten
# Überprüfe, ob d zu groß:
if d > (math.log2(n)**2):
    raise ValueError("Ungültiger Graph-Typ")
else:
    # Obergrenze berechnen n⋅d⋅(log(n)^2)
    upper_bound = 50
    # upper_bound = int(n * d * (math.log2(n))**2)

    # Initialisieren:
    graphs = [copy.deepcopy(G)]
    cut_strains = []
    expected_cut_strains = []
    #03.09.25
    expected_strains_alternative = []
    #cheeger_constants = []
    cut_sizes = []
    conductances = []
    cut_edges_list = []
    flip_info = []

    # Erste Berechnung
    strain, conductance, cut_edges = Calc_updated.cut_metrics(G, cut_set, d)
    cut_size = len(cut_edges)
    expected_strain = Calc_updated.expected_cut_strain_exact(G, cut_set, d, strain)
    #03.09.25
    expected_strain_alternative = Calc_updated.calculate_expected_cut_strain_alternative(G, cut_set, d, strain)

    # strain, conductance, cut_edges = Calc_updated.cut_metrics(G, cut_set, d)
    cut_strains.append(strain)
    expected_cut_strains.append(expected_strain)
    #03.09.25
    expected_strains_alternative.append(expected_strain_alternative)
    cut_sizes.append(cut_size)
    conductances.append(conductance)
    # cheeger_constants.append(cut_size)
    cut_edges_list.append(cut_edges)
    flip_info.append((set(), set()))

    current_G = G  # nötig??
    number_flips = upper_bound
    while len(graphs) <= number_flips:
        new_G, removed, added = Graph.flip_operation(current_G)
        if removed is None or added is None:
            continue
        current_G = new_G
        graphs.append(copy.deepcopy(current_G))
        strain, conductance, cut_edges = Calc_updated.cut_metrics(current_G, cut_set, d)
        expected_strain = Calc_updated.expected_cut_strain_exact(current_G, cut_set, d, strain)
        #03.09.25
        expected_strain_alternative = Calc_updated.calculate_expected_cut_strain_alternative(current_G, cut_set, d, strain)
        cut_size = len(cut_edges)

        cut_strains.append(strain)
        expected_cut_strains.append(expected_strain)
        #03.09.25
        expected_strains_alternative.append(expected_strain_alternative)
        cut_sizes.append(cut_size)
        conductances.append(conductance)
        cut_edges_list.append(cut_edges)
        flip_info.append((removed, added))

    expected_cut_strains = [cut_strains[0]] + expected_cut_strains[:-1]

    # Interaktive Ansicht mit 2 Achsen
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])
    ax_cutsize = fig.add_subplot(gs[0, 0])
    ax_strain = fig.add_subplot(gs[1, 0])
    ax_graph = fig.add_subplot(gs[:, 1])
    plt.subplots_adjust(bottom=0.2, hspace=0.4, wspace=0.3)
    current_step = [0]

    def draw_flip_step(index):
        ax_cutsize.clear()
        ax_strain.clear()
        ax_graph.clear()

        # Plot 1: Cut Size
        ax_cutsize.plot(range(len(cut_sizes)), cut_sizes,
                        label="Cut Size", color='orange')
        ax_cutsize.plot(range(len(conductances)), conductances,
                        label="Conductance", color='green')
        ax_cutsize.axvline(index, color='gray', linestyle='--')
        ax_cutsize.axvline(upper_bound, color='red',
                           linestyle=':', label='Upper Bound')
        ax_cutsize.set_ylabel("Cut Metrics")
        ax_cutsize.set_xlabel("Flip-Schritte")
        ax_cutsize.set_title("Entwicklung von Cut Size und Conductance")
        ax_cutsize.legend()
        ax_cutsize.grid(True)
        '''
        ax_cutsize.plot(range(len(cheeger_constants)), cheeger_constants,
                        label="Cut Size", color='orange')
        ax_cutsize.axvline(index, color='gray', linestyle='--')
        ax_cutsize.axvline(upper_bound, color='red', linestyle=':', label='Upper Bound')
        ax_cutsize.set_title("Cut Size")
        ax_cutsize.set_ylabel("Wert")
        ax_cutsize.set_xlabel("Flip-Schritte")
        
        ax_cutsize.grid(True)
        ax_cutsize.legend()
        '''

        # Plot 2: Strain and Expected Strain
        ax_strain.plot(
            range(len(cut_strains)), cut_strains,
            label="Cut Strain", color='blue')
        # ax_strain.plot(range(len(expected_cut_strains)), expected_cut_strains, label="Expected Cut Strain", color='purple')
        # Expected Strain um eins versetzt
        ax_strain.plot(
            range(len(expected_cut_strains)),
            expected_cut_strains,
            label="Expected Cut Strain (nach Flip)",
            color='purple'
        )
        # 03.09.25 Expected Strain Alternative
        ax_strain.plot(
        range(1, len(expected_strains_alternative)),
        expected_strains_alternative[:-1],
        label="Expected Cut Strain Alternative (nach Flip)",
        color='green'
        )
        ax_strain.axvline(index, color='gray', linestyle='--')
        ax_strain.axvline(upper_bound, color='red',
                          linestyle=':', label='Upper Bound')
        ax_strain.set_ylabel("Strain")
        ax_cutsize.set_xlabel("Flip-Schritte")
        ax_strain.set_title(
            "Entwicklung von Cut Strain und Expected Cut Strain")
        ax_strain.legend()
        ax_strain.grid(True)
        '''
        ax_strain.plot(range(len(cut_strains)), cut_strains,
                       label="Cut Strain", color='blue')
        ax_strain.plot(range(len(expected_cut_strains)), expected_cut_strains,
                       label="Expected Cut Strain", color='purple')
        ax_strain.axvline(index, color='gray', linestyle='--')
        ax_strain.axvline(upper_bound, color='red', linestyle=':', label='Upper Bound')
        ax_strain.set_title("Cut Strain vs. Expected")
        ax_strain.set_ylabel("Wert")
        ax_strain.set_xlabel("Flip-Schritte")
    
        ax_strain.grid(True)
        ax_strain.legend()
        '''
        # Graph Darstellung
        G = graphs[index]
        cut_edges = cut_edges_list[index]
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

        nx.draw(
            G,
            pos,
            ax=ax_graph,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=500,
            font_size=8
        )

        if index == 0:
            ecs_display = "N/A"  # Keine ECS vor der ersten Flip-Operation
        else:
            # ecs_display = f"{expected_cut_strains[index - 1]:.3f}"
            ecs_display = f"expected_strain: {expected_cut_strains[index - 1]:.3f}, alternative: {expected_strains_alternative[index - 1]:.3f}"

        ax_graph.set_title(
            f"Flip-Schritt {index}: \n"
            f"d =  {d}, n = {n}, Upper Bound = {upper_bound} \n"
            f"Cut-Strain: {cut_strains[index]:.3f}, Expected Cut-Strain (nach Flip): {ecs_display}, "
            f"Cut-Size: {cut_sizes[index]},Conductance: {conductances[index]} \n"
            f"Removed Edges: {removed_edges}, Added Edges: {added_edges}"
        )
        plt.draw()

    def next_step(event):
        current_step[0] = (current_step[0] + 1) % len(graphs)
        draw_flip_step(current_step[0])

    def prev_step(event):
        current_step[0] = (current_step[0] - 1) % len(graphs)
        draw_flip_step(current_step[0])

    axprev = plt.axes([0.25, 0.02, 0.15, 0.05])
    axnext = plt.axes([0.60, 0.02, 0.15, 0.05])
    bprev = Button(axprev, '⟵ Previous')
    bnext = Button(axnext, 'Next ⟶')
    bprev.on_clicked(prev_step)
    bnext.on_clicked(next_step)

    draw_flip_step(current_step[0])
    plt.show()
