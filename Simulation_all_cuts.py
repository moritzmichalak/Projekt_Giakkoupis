import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import copy
from collections import OrderedDict
import numpy as np

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

# --- Nutzerentscheidung: Graph visualisieren? ---
default_visualize = (n <= 100)
default_str = "y" if default_visualize else "n"
print(f"Visualize the graph during navigation? (y/n) [default {default_str}]")
resp = input().strip().lower()
visualize_graph = (default_visualize if resp == "" else resp in ("y", "yes", "j", "ja"))
draw_labels = visualize_graph and (n <= 100)

# --- NEU: Intervall für Actual/Expected (Giakkoupis) ---
def _get_int_or_default(prompt: str, default_val: int) -> int:
    print(prompt)
    try:
        v = int(input())
        return max(1, v)
    except Exception:
        print(f"Invalid input, using default = {default_val}.")
        return default_val

actual_period = _get_int_or_default(
    "Every how many steps should the ACTUAL Cut Strain be recomputed? (e.g., 1 = every step)",
    1
)
exact_period = _get_int_or_default(
    "Every how many steps should the EXPECTED Cut Strain (Giakkoupis) be recomputed? (e.g., 1 = every step)",
    1
)

# --- Alternative Expected Cut Strain optional ---
print("Compute alternative Expected Cut Strain? (y/n) [default n]")
resp_alt = input().strip().lower()
compute_alt = (resp_alt in ("y", "yes", "j", "ja"))
alt_period = None
if compute_alt:
    alt_period = _get_int_or_default(
        "Every how many steps should the ALTERNATIVE metric be recomputed? (e.g., 50)",
        50
    )

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
# EIN Simulationslauf + Metriken on-the-fly (ohne Graph-Kopien)
# =========================
graphs = [G0.copy()] if visualize_graph else []  # Snapshot nur für Visualisierung
flip_info = [(set(), set())]                     # Index 0 = Startzustand
changed_flags = [False]                          # Schritt 0: nichts geändert
current_G = G0

# Metrik-Container: Schritt 0 initialisieren
metrics = {
    name: {"strain": [], "exp_exact": [], "exp_alt": [], "exp_alt_last_step": []}
    for name in CUTS
}

# Schritt 0 (vor erster Flipoperation)
for name, S in CUTS.items():
    s0, _, _ = Calc_updated.cut_metrics(current_G, S, d)
    metrics[name]["strain"].append(s0)
    metrics[name]["exp_exact"].append(np.nan)       # Expected ab Schritt 1
    metrics[name]["exp_alt"].append(np.nan)         # Alt optional/periode
    metrics[name]["exp_alt_last_step"].append(None)

amount_flip_operations = 0
# Hauptschleife: Schritte 1..upper_bound
for t in range(1, upper_bound + 1):
    amount_flip_operations += 1
    # --- Für EXPECTED_exact(t) brauchen wir ACTUAL(t-1) auf dem VORHERigen Graphen
    #     Nur für die Cuts berechnen, bei denen Expected jetzt fällig ist.
    prev_strain_for_exact = {}
    for name, S in CUTS.items():
        if (t % exact_period) == 0:
            s_prev, _, _ = Calc_updated.cut_metrics(current_G, S, d)
            prev_strain_for_exact[name] = s_prev

    # --- Flip durchführen (mutiert current_G nur bei Erfolg)
    current_G, removed, added = Graph.flip_operation(current_G, number_of_cliques, size_of_cliques)
    changed = (removed is not None and added is not None)

    # Visualisierung: aktuellen Snapshot speichern
    if visualize_graph:
        graphs.append(current_G.copy())

    # Flip-Info pflegen
    if changed:
        removed_norm = {tuple(sorted(e)) for e in removed}
        added_norm   = {tuple(sorted(e)) for e in added}
        flip_info.append((removed_norm, added_norm))
        changed_flags.append(True)
    else:
        flip_info.append((set(), set()))
        changed_flags.append(False)

    # --- Metriken für diesen Schritt t
    for name, S in CUTS.items():
        if not changed:
            # Keine Änderungen: Werte übernehmen
            metrics[name]["strain"].append(metrics[name]["strain"][t-1])
            metrics[name]["exp_exact"].append(metrics[name]["exp_exact"][t-1])
            metrics[name]["exp_alt"].append(metrics[name]["exp_alt"][t-1])
            metrics[name]["exp_alt_last_step"].append(metrics[name]["exp_alt_last_step"][t-1])
            continue

        # Fälligkeitslogik
        need_actual_t = (t % actual_period == 0)
        need_exact_t  = (t % exact_period  == 0)
        need_alt_t    = (compute_alt and (t % alt_period == 0))

        # ACTUAL(t) nur berechnen, wenn für ACTUAL / EXPECTED / ALT benötigt
        strain_t = None
        if need_actual_t or need_exact_t or need_alt_t:
            strain_t, _, _ = Calc_updated.cut_metrics(current_G, S, d)

        # ACTUAL(t) in Zeitreihe schreiben
        if need_actual_t or need_exact_t:
            metrics[name]["strain"].append(
                strain_t if strain_t is not None else metrics[name]["strain"][t-1]
            )
        else:
            metrics[name]["strain"].append(metrics[name]["strain"][t-1])

        # EXPECTED_exact(t): nutzt den frisch berechneten ACTUAL(t-1) vom Vor-Graphen
        if need_exact_t:
            # prev_strain_for_exact[name] ist oben (vor dem Flip) berechnet
            exp_exact_t = Calc_updated.expected_cut_strain_exact(
                current_G, S, d, prev_strain_for_exact[name]
            )
            metrics[name]["exp_exact"].append(exp_exact_t)
        else:
            metrics[name]["exp_exact"].append(metrics[name]["exp_exact"][t-1])

        # EXPECTED_alt(t): optional + periodisch, nutzt ACTUAL(t)
        if need_alt_t:
            if strain_t is None:
                strain_t, _, _ = Calc_updated.cut_metrics(current_G, S, d)
            exp_alt_t = Calc_updated.calculate_expected_cut_strain_alternative(current_G, S, d, strain_t)
            metrics[name]["exp_alt"].append(exp_alt_t)
            metrics[name]["exp_alt_last_step"].append(t)
        else:
            metrics[name]["exp_alt"].append(metrics[name]["exp_alt"][t-1])
            metrics[name]["exp_alt_last_step"].append(metrics[name]["exp_alt_last_step"][t-1])
        # Fortschritt alle ~1% (mind. alle 1 Versuche)
    report_every = max(1, upper_bound // 100)
    if amount_flip_operations % report_every == 0:
        percentage = (amount_flip_operations * 100.0) / upper_bound
        print(f"{percentage:.0f} %  |  attempts: {amount_flip_operations}")
# Schritte für die Plots (aus der Metrik-Länge ablesen, unabhängig von Graph-Snapshots)
steps = list(range(len(next(iter(metrics.values()))["strain"])))  # 0..T
total_steps = len(steps) - 1

    
# =========================
# Plot-Layout: links 3 Charts (pro Cut), optional rechts Graph; Info oben
# =========================
fig = plt.figure(figsize=(14, 9))

if visualize_graph:
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
else:
    gs = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[0.45, 1.0, 1.0, 1.0],
        left=0.06, right=0.98, bottom=0.10, top=0.95,
        hspace=0.25
    )
    # Info oben, darunter die drei Charts – volle Breite
    ax_info  = fig.add_subplot(gs[0, 0])
    ax_cut1  = fig.add_subplot(gs[1, 0])
    ax_cut2  = fig.add_subplot(gs[2, 0], sharex=ax_cut1)
    ax_cut3  = fig.add_subplot(gs[3, 0], sharex=ax_cut1)
    ax_graph = None  # signalisiert "kein Graph zeichnen"

# --- Helper zum Plotten (nur 2 Kurven je Chart) ---
def plot_cut_metrics(ax, name, show_xlabel=False):
    y_strain = metrics[name]["strain"]
    y_exact  = metrics[name]["exp_exact"]

    # Erwartete Werte bei Schritt 0 im Plot ausblenden (Linie startet bei 1)
    y_exact_plot = np.array(y_exact, dtype=float)
    if len(y_exact_plot) > 0:
        y_exact_plot[0] = np.nan

    l1, = ax.plot(steps, y_strain,     label="Actual Cut Strain")
    l2, = ax.plot(steps, y_exact_plot, label="Expected Cut Strain (Giakkoupis)")
    ax.set_title(CUT_TITLES.get(name, name))
    ax.grid(True, alpha=0.3)
    if len(steps) > 1:
        ax.set_xlim(0, len(steps) - 1)
    else:
        ax.set_xlim(-0.5, 0.5)  # vermeidet die Warnung bei nur einem Punkt

    if show_xlabel:
        ax.set_xlabel("Flip-Index")
    return l1, l2  # nur 2 Handles

# --- Plots setzen ---
lines_top = plot_cut_metrics(ax_cut1, "biggest")         # Handles für Legende
_ = plot_cut_metrics(ax_cut2, f"block_{k_block}")
_ = plot_cut_metrics(ax_cut3, "every_second", show_xlabel=True)

ax_cut2.set_ylabel("Wert")

# ---- NUR EINE LEGENDE: figure-weit oben ----
fig.legend(
    handles=lines_top,
    labels=[ln.get_label() for ln in lines_top],
    loc="upper center", bbox_to_anchor=(0.5, 0.995),
    ncol=2, frameon=False
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
axbox = plt.axes([0.08, 0.02, 0.18, 0.05])
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
    if not visualize_graph or ax_graph is None:
        return  # Graph-Darstellung deaktiviert
    
    ax_graph.clear()
    G = graphs[i]
    removed_edges, added_edges = flip_info[i] if i < len(flip_info) else (set(), set())

    # Knotenfarben: ausgewählten Cut gelb markieren
    S = CUTS[selected_cut_key[0]]
    node_cols = ["yellow" if (u in S) else "lightblue" for u in G.nodes()]

    # Basis: Knoten + vorhandene Kanten
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_cols,
                           node_size=80 if len(G) > 150 else 120)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color="black", width=1.2)

    # Hinzugefügte Kanten in diesem Schritt (falls im aktuellen G vorhanden): rot
    added_list = [(u, v) for (u, v) in added_edges if G.has_edge(u, v)]
    if added_list:
        nx.draw_networkx_edges(G, pos, edgelist=added_list, ax=ax_graph,
                               edge_color="red", width=2.0)

    # Entfernte Kanten in diesem Schritt (existieren nicht mehr in G): gestrichelt blau
    for (u, v) in removed_edges:
        if u in pos and v in pos:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax_graph.plot(x, y, linestyle="--", color="blue", linewidth=1.6, alpha=0.9, zorder=0)

    # Labels optional (bei großen n aus)
    if draw_labels:
        labels_nodes = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels_nodes, ax=ax_graph, font_size=6)

    ax_graph.set_title(f"Ring of Cliques – Schritt {i}/{total_steps}")
    #ax_graph.set_title(f"Ring of Cliques – Schritt {i}/{len(graphs)-1}")
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

    # Expected (exact): bei Schritt 0 "N/A"
    if i == 0 or not np.isfinite(metrics[k]["exp_exact"][i]):
        val_exact_str = "N/A"
    else:
        val_exact_str = f"{metrics[k]['exp_exact'][i]:.4f}"

    # Alternative: Wert + letzter Berechnungsschritt (oder deaktiviert)
    last_alt_step = metrics[k]["exp_alt_last_step"][i]
    if not compute_alt:
        val_alt_str = "N/A"
        alt_suffix = "(deaktiviert)"
    else:
        if not np.isfinite(metrics[k]["exp_alt"][i]):
            val_alt_str = "N/A"
        else:
            val_alt_str = f"{metrics[k]['exp_alt'][i]:.4f}"

        if last_alt_step is None:
            alt_suffix = "(noch nicht berechnet)"
        else:
            alt_suffix = f"(zuletzt berechnet bei Schritt {last_alt_step})"

    lines = [
        f"Schritt {i}/{total_steps}   |   Ausgewählter Cut: {title}",
        f"Actual Cut Strain: {val_strain:.4f}  (Interval: {actual_period})",
        f"Expected Cut Strain (Giakkoupis): {val_exact_str}  (Interval: {exact_period})",
        f"Expected Cut Strain (alternative): {val_alt_str} {alt_suffix}" + (f"  (Period: {alt_period})" if compute_alt else ""),
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
    # Info-Panel + (optional) Graph neu zeichnen
    update_info_panel(i)
    if ax_graph is not None and visualize_graph:
        draw_graph_at_step(i)
    fig.canvas.draw_idle()

def on_next(event):
    current_step[0] = (current_step[0] + 1) % len(steps)
    update_all(current_step[0])

def on_prev(event):
    current_step[0] = (current_step[0] - 1) % len(steps)
    update_all(current_step[0])

def on_submit_step(text):
    # Eingabe verarbeiten und zum Schritt springen
    try:
        x = int(text)
    except Exception:
        return  # ungültig -> ignoriere
    x = max(0, min(len(steps) - 1, x))
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
