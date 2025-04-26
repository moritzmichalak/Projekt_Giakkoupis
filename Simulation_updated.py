import networkx as nx
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons
import matplotlib.pyplot as plt
import copy
from matplotlib.widgets import Button
import Graph
import Calc_updated

# Wähle Art des Graphen (Ring of Cliques, random ,... TO DO)
type_of_graph = "rof"
#type_of_graph = "random"
if type_of_graph == "rof":
    G, pos = Graph.create_ring_of_cliques(5, 5)
    cut_set = {"C1_4", "C1_3", "C1_2", "C1_1", "C1_0"}
    # cut_set = {"C0_2", "C0_3"}
    d = 4
elif type_of_graph == "random":
    G, pos = Graph.create_random_d_regular_graph()
    cut_set = Graph.generate_random_cut(G)
    d = Calc_updated.calculate_d(G)
else: 
    print("Wähl einen gültigen Graph aus: {rof, random}")

graphs = [copy.deepcopy(G)]
cut_strains = []
cheeger_constants = []
cut_edges_list = []
flip_info = []
expected_cut_strains = []
strain, conductance, cut_edges = Calc_updated.cut_metrics(G, cut_set, d)
cut_size = len(cut_edges)
cut_strains.append(strain)
cheeger_constants.append(cut_size)
cut_edges_list.append(cut_edges)
flip_info.append((set(), set()))
current_G = G
number_flips = 400
while len(graphs) <= number_flips:
    new_G, removed, added = Graph.flip_operation(current_G)
    if removed is None or added is None:
        continue
    current_G = new_G
    graphs.append(copy.deepcopy(current_G))
    strain, conductance, cut_edges = Calc_updated.cut_metrics(current_G, cut_set, d)
    # strain, cut_edges = Calc_updated.cut_metrics(current_G, cut_set)
    cut_size = len(cut_edges)
    cut_strains.append(strain)

    expected_strain = Calc_updated.expected_cut_strain_exact(current_G, cut_set, d)
    expected_cut_strains.append(expected_strain)

    cheeger_constants.append(cut_size)
    cut_edges_list.append(cut_edges)
    flip_info.append((removed, added))

expected_strain = Calc_updated.expected_cut_strain_exact(current_G, cut_set, d)
expected_cut_strains.append(expected_strain)

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
    ax_cutsize.plot(range(len(cheeger_constants)), cheeger_constants,
                    label="Cut Size", color='orange')
    ax_cutsize.axvline(index, color='gray', linestyle='--')
    ax_cutsize.set_title("Cut Size")
    ax_cutsize.set_ylabel("Wert")
    ax_cutsize.set_xlabel("Flip-Schritte")
    ax_cutsize.grid(True)
    ax_cutsize.legend()

    # Plot 2: Strain and Expected Strain
    ax_strain.plot(range(len(cut_strains)), cut_strains,
                   label="Cut Strain", color='blue')
    ax_strain.plot(range(len(expected_cut_strains)), expected_cut_strains,
                   label="Expected Cut Strain", color='purple')
    ax_strain.axvline(index, color='gray', linestyle='--')
    ax_strain.set_title("Cut Strain vs. Expected")
    ax_strain.set_ylabel("Wert")
    ax_strain.set_xlabel("Flip-Schritte")
    ax_strain.grid(True)
    ax_strain.legend()

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
    ax_graph.set_title(
        f"Flip-Schritt {index}: \n"
        f"Cut-Strain: {cut_strains[index]:.3f}, "
        f"Expected Cut-Strain: {expected_cut_strains[index]:.3f}, \n"
        f"Cut-Size: {cheeger_constants[index]}, \n"
        f"Conductance: {conductance}, \n"
        f"Removed Edges: {removed_edges}, \n"
        f"Added Edges: {added_edges}"
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
