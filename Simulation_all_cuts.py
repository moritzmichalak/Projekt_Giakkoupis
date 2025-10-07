import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox  # <-- TextBox hinzugefügt
import copy
from collections import OrderedDict
import numpy as np  # <-- NEU: für NaN-Maskierung der Expected-Kurven

import Graph
import Calc_updated

# =========================
# Eingaben
# =========================
print("Choose number of nodes:")
n = int(input())

possible_d = Calc_updated.calculate_possible_d(n)
if not possible_d:
    raise ValueError("No valid degrees d>2 with n divisible by (d+1). Try another n.")

print("Choose degree from ", possible_d)
d = int(input())
if d not in possible_d:
    raise ValueError(f"d={d} is not valid for n={n}. Valid: {possible_d}")

number_of_cliques = n // (d + 1)
size_of_cliques = d + 1
print(f"You chose a graph with {number_of_cliques} cliques, each with a size of {size_of_cliques} nodes.")

G0, pos = Graph.create_ring_of_cliques(number_of_cliques, size_of_cliques)

print(f"How many adjacent cliques should be in the block-cut? Choose k from 1 to {number_of_cliques}:")
k_block = int(input())
if not (1 <= k_block <= number_of_cliques):
    raise ValueError(f"k must be in [1, {number_of_cliques}]")

print("How many flip operations do you want to run?")
upper_bound = int(input())
if upper_bound < 0:
    raise ValueError("upper_bound must be >= 0")

# =========================
# Drei Cuts (in gewünschter Reihenfolge)
# =========================
CUTS = OrderedDict([
    ("biggest",        Graph.generate_cut_3(G0, number_of_cliques, size_of_cliques)),
    (f"block_{k_block}", Graph.generate_cut_2(G0, k_block, size_of_cliques)),
    ("every_second",   Graph.generate_cut_1(G0)),
])

CUT_TITLES = {
    "biggest": "Biggest Cut",
    f"block_{k_block}": f"Block-Cut (k = {k_block})",
    "every_second": "Jede zweite Clique",
}

# =========================
# EIN Simulationslauf: Graph-Snapshots + Flip-Infos
# =========================
graphs = [copy.deepcopy(G0)]
flip_info = [(set(), set())]  # i=0 leer
current_G = G0

while len(graphs) <= upper_bound:
    new_G, removed, added = Graph.flip_operation(current_G)
    if removed is None or added is None:
        continue  # ungültiger Versuch -> erneut
    current_G = new_G
    graphs.append(copy.deepcopy(current_G))
    removed_norm = {tuple(sorted(e)) for e in removed}
    added_norm   = {tuple(sorted(e)) for e in added}
    flip_info.append((removed_norm, added_norm))

steps = list(range(len(graphs)))

# =========================
# Metriken für alle Cuts und Snapshots
# =========================

# Expected-Arrays mit 0 für Schritt 0 starten (für Konsistenz),
# aber die Anzeige in den Plots maskieren wir später auf NaN.
metrics = {
    name: {"strain": [], "exp_exact": [0.0], "exp_alt": [0.0]}
    for name in CUTS
}

# Schritt 0: nur den tatsächlichen Strain berechnen
G0snap = graphs[0]
for name, S in CUTS.items():
    strain0, _, _ = Calc_updated.cut_metrics(G0snap, S, d)
    metrics[name]["strain"].append(strain0)

# Ab Schritt 1: Strain + Expected-Werte anhängen
for Gsnap in graphs[1:]:
    for name, S in CUTS.items():
        strain, _, _ = Calc_updated.cut_metrics(Gsnap, S, d)
        exp_exact = Calc_updated.expected_cut_strain_exact(Gsnap, S, d, strain)
        exp_alt   = Calc_updated.calculate_expected_cut_strain_alternative(Gsnap, S, d, strain)
        metrics[name]["strain"].append(strain)
        metrics[name]["exp_exact"].append(exp_exact)
        metrics[name]["exp_alt"].append(exp_alt)

# =========================
# Plot-Layout: links 3 Charts (pro Cut), rechts Info+Graph (50%)
# =========================
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(
    nrows=4, ncols=2,
    width_ratios=[1.0, 1.0],              # 50% Charts links, 50% rechts (Info + Graph)
    height_ratios=[0.50, 1.0, 1.0, 1.0],  # schlankes Info-Panel oben
    left=0.05, right=0.98, bottom=0.10, top=0.95,
    wspace=0.25, hspace=0.25
)

# Links: drei Charts (je Cut)
ax_cut1  = fig.add_subplot(gs[1, 0])
ax_cut2  = fig.add_subplot(gs[2, 0], sharex=ax_cut1)
ax_cut3  = fig.add_subplot(gs[3, 0], sharex=ax_cut1)
# Rechts: Info-Panel oben + Graph unten (über 3 Zeilen)
ax_info  = fig.add_subplot(gs[0, 1])
ax_graph = fig.add_subplot(gs[1:, 1])

# --- Helper zum Plotten der 3 Kennzahlen pro Cut ---
def plot_cut_metrics(ax, name, show_xlabel=False):
    y_strain = metrics[name]["strain"]
    y_exact  = metrics[name]["exp_exact"]
    y_alt    = metrics[name]["exp_alt"]

    # Erwartete Werte bei Schritt 0 im Plot ausblenden (Linien starten bei 1)
    y_exact_plot = np.array(y_exact, dtype=float)
    y_alt_plot   = np.array(y_alt, dtype=float)
    if len(y_exact_plot) > 0:
        y_exact_plot[0] = np.nan
    if len(y_alt_plot) > 0:
        y_alt_plot[0] = np.nan

    l1, = ax.plot(steps, y_strain,      label="Actual Cut Strain")
    l2, = ax.plot(steps, y_exact_plot,  label="Expected Cut Strain (Giakkoupis)")
    l3, = ax.plot(steps, y_alt_plot,    label="Expected Cut Strain (alternative)")
    ax.set_title(CUT_TITLES.get(name, name))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(steps) - 1)
    if show_xlabel:
        ax.set_xlabel("Flip-Index")
    return l1, l2, l3  # <-- Linien-Handles zurückgeben

# --- Plots setzen ---
lines_top = plot_cut_metrics(ax_cut1, "biggest")         # <-- Handles für Legende
_ = plot_cut_metrics(ax_cut2, f"block_{k_block}")
_ = plot_cut_metrics(ax_cut3, "every_second", show_xlabel=True)

ax_cut2.set_ylabel("Wert")

# ---- NUR EINE LEGENDE: figure-weit oben (statt dreimal) ----
fig.legend(
    handles=lines_top,
    labels=[ln.get_label() for ln in lines_top],
    loc="upper center", bbox_to_anchor=(0.5, 0.995),
    ncol=3, frameon=False
)

# Cursor-Linien (aktueller Schritt) in allen drei Charts
cursor1 = ax_cut1.axvline(x=0, linestyle="--", color="grey", linewidth=1)
cursor2 = ax_cut2.axvline(x=0, linestyle="--", color="grey", linewidth=1)
cursor3 = ax_cut3.axvline(x=0, linestyle="--", color="grey", linewidth=1)

# =========================
# Info-Panel + Dropdown
# =========================
ax_info.set_axis_off()

def format_edges(edges_set):
    if not edges_set:
        return "-"
    try:
        seq = sorted(list(edges_set))
    except TypeError:
        seq = list(edges_set)
    return ", ".join([f"({u},{v})" for (u, v) in seq])

info_bbox = ax_info.get_position()

# Rechts im Info-Panel: Dropdown
drop_ax = fig.add_axes([
    info_bbox.x0 + 0.60 * info_bbox.width,
    info_bbox.y0 + 0.20 * info_bbox.height,
    0.39 * info_bbox.width,
    0.60 * info_bbox.height
])

# Zusätzlich unten: TextBox "Go to step"
# (Platzierung neben den Buttons, leicht nach links versetzt)
axbox = plt.axes([0.08, 0.02, 0.18, 0.05])  # <-- Position/Größe der TextBox
step_box = TextBox(axbox, 'Go to step: ', initial='0')

# Dropdown-Optionen
label_by_key = {k: CUT_TITLES.get(k, k) for k in CUTS.keys()}
key_by_label = {v: k for k, v in label_by_key.items()}
labels = [label_by_key[k] for k in CUTS.keys()]
selected_cut_key = [list(CUTS.keys())[0]]  # initial: "biggest"

dropdown = None
def _bind_dropdown(ax, labels, initial_label):
    global dropdown
    try:
        from matplotlib.widgets import Dropdown as MplDropdown
        dropdown = MplDropdown(ax, label='Cut', options=labels, value=initial_label)
        dropdown.on_changed(on_cut_changed)
        return "dropdown"
    except Exception:
        from matplotlib.widgets import RadioButtons
        ax.clear()
        dropdown = RadioButtons(ax, labels, active=labels.index(initial_label))
        dropdown.on_clicked(on_cut_changed)
        return "radio"

def on_cut_changed(label):
    if label in key_by_label:
        selected_cut_key[0] = key_by_label[label]
    else:
        selected_cut_key[0] = key_by_label.get(str(label), selected_cut_key[0])
    update_all(current_step[0])

_ui_type = _bind_dropdown(drop_ax, labels, label_by_key[selected_cut_key[0]])

# =========================
# Graph-Zeichnen
# =========================
def draw_graph_at_step(i: int):
    ax_graph.clear()
    G = graphs[i]
    removed_edges, added_edges = flip_info[i] if i < len(flip_info) else (set(), set())

    # Knotenfarben: ausgewählten Cut gelb markieren
    S = CUTS[selected_cut_key[0]]
    node_cols = ["yellow" if (u in S) else "lightblue" for u in G.nodes()]

    # Basis: Knoten + vorhandene Kanten
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_cols, node_size=120)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color="black", width=1.2)

    # Hinzugefügte Kanten in diesem Schritt (falls im aktuellen G vorhanden): rot
    added_list = [(u, v) for (u, v) in added_edges if G.has_edge(u, v)]
    if added_list:
        nx.draw_networkx_edges(G, pos, edgelist=added_list, ax=ax_graph, edge_color="red", width=2.0)

    # Entfernte Kanten in diesem Schritt (existieren nicht mehr in G): gestrichelt blau
    for (u, v) in removed_edges:
        if u in pos and v in pos:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax_graph.plot(x, y, linestyle="--", color="blue", linewidth=1.8, alpha=0.9, zorder=0)

    # Knotennamen (Labels) – transparenter Hintergrund
    labels_nodes = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels_nodes, ax=ax_graph, font_size=6)

    ax_graph.set_title(f"Ring of Cliques – Schritt {i}/{len(graphs)-1}")
    ax_graph.set_axis_off()

# =========================
# Info-Panel aktualisieren
# =========================
def update_info_panel(i: int):
    ax_info.clear()
    ax_info.set_axis_off()
    removed_edges, added_edges = flip_info[i] if i < len(flip_info) else (set(), set())
    k = selected_cut_key[0]
    title = CUT_TITLES.get(k, k)

    val_strain = metrics[k]["strain"][i]
    # Bei Schritt 0 Expected-Werte als "N/A" anzeigen
    if i == 0:
        val_exact_str = "N/A"
        val_alt_str   = "N/A"
    else:
        val_exact_str = f"{metrics[k]['exp_exact'][i]:.4f}"
        val_alt_str   = f"{metrics[k]['exp_alt'][i]:.4f}"

    lines = [
        f"Schritt {i}/{len(graphs)-1}   |   Ausgewählter Cut: {title}",
        f"Actual Cut Strain: {val_strain:.4f}",
        f"Expected Cut Strain (Giakkoupis): {val_exact_str}",
        f"Expected Cut Strain (alternative): {val_alt_str}",
        f"Removed: {format_edges(removed_edges)}",
        f"Added:   {format_edges(added_edges)}",
    ]
    ax_info.text(0.01, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10)

# =========================
# Navigation + Direkt-Sprung per TextBox
# =========================
current_step = [0]

def update_all(i: int):
    # Cursor updaten
    for ln in (cursor1, cursor2, cursor3):
        ln.set_xdata([i, i])
    # Info-Panel + Graph neu zeichnen
    update_info_panel(i)
    draw_graph_at_step(i)
    fig.canvas.draw_idle()

def on_next(event):
    current_step[0] = (current_step[0] + 1) % len(graphs)
    update_all(current_step[0])

def on_prev(event):
    current_step[0] = (current_step[0] - 1) % len(graphs)
    update_all(current_step[0])

def on_submit_step(text):
    # Eingabe verarbeiten und zum Schritt springen
    try:
        x = int(text)
    except Exception:
        return  # ungültig -> ignoriere
    x = max(0, min(len(graphs) - 1, x))  # clamp
    current_step[0] = x
    update_all(x)

# Buttons unten (zentriert)
axprev = plt.axes([0.30, 0.02, 0.16, 0.05])
axnext = plt.axes([0.54, 0.02, 0.16, 0.05])
bprev = Button(axprev, '⟵ Previous')
bnext = Button(axnext, 'Next ⟶')
bprev.on_clicked(on_prev)
bnext.on_clicked(on_next)

# TextBox-Callback registrieren
step_box.on_submit(on_submit_step)

# Initial zeichnen
update_all(current_step[0])
plt.show()
