import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
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

# --- Intervall-/Eingabe-Helfer ---
def _get_int_or_default(prompt: str, default_val: int) -> int:
    print(prompt)
    try:
        v = int(input())
        return max(1, v)
    except Exception:
        print(f"Invalid input, using default = {default_val}.")
        return default_val

def _get_float_or_default(prompt: str, default_val: float) -> float:
    print(prompt)
    s = input().strip()
    if s == "":
        return default_val
    try:
        return float(s)
    except Exception:
        print(f"Invalid input, using default = {default_val}.")
        return default_val

# --- Actual (immer aktiv, aber periodisch) ---
actual_period = _get_int_or_default(
    "Every how many steps should the ACTUAL Cut Strain be recomputed? (e.g., 1 = every step)",
    1
)

# --- Expected (Giakkoupis) optional ---
print("Compute EXPECTED Cut Strain (Giakkoupis)? (y/n) [default y]")
resp_exact = input().strip().lower()
compute_exact = not (resp_exact in ("n", "no", "nein"))
if compute_exact:
    exact_period = _get_int_or_default(
        "Every how many steps should the EXPECTED Cut Strain (Giakkoupis) be recomputed? (e.g., 1 = every step)",
        1
    )
else:
    exact_period = None  # deaktiviert

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

# --- δ für die Expected Cut Size Schranke (Default 1/d) ---
delta_cut_size  = 1.0 / d

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

# --- Smallest-Cut bestimmen (für LB-Prüfung)
if "block_1" in CUTS:
    LB_TRACK_KEY = "block_1"
else:
    init_sizes = {name: len(Calc_updated.cut_metrics(G0, S, d)[2]) for name, S in CUTS.items()}
    LB_TRACK_KEY = min(init_sizes, key=init_sizes.get)

# Initiale Cut Size und Eligibility (nur wenn c0 <= d wird LB geprüft)
c0_init = len(Calc_updated.cut_metrics(G0, CUTS[LB_TRACK_KEY], d)[2])
LB_ELIGIBLE = (c0_init <= d)

# =========================
# EIN Simulationslauf + Metriken on-the-fly (ohne Graph-Kopien)
# =========================
graphs = [G0.copy()] if visualize_graph else []  # Snapshot nur für Visualisierung
flip_info = [(set(), set())]                     # Index 0 = Startzustand
changed_flags = [False]                          # Schritt 0: nichts geändert
hub_edges = [None]                               # Schritt 0: keine Hub-Kante
current_G = G0

# Metrik-Container
metrics = {
    name: {"strain": [], "exp_exact": [], "exp_alt": [], "exp_alt_last_step": [],
           "cut_size": [], "exp_cut_size_lb": [], "under_lb_flag": []}
    for name in CUTS
}
# für LB-Cut zusätzlich:
metrics[LB_TRACK_KEY]["lb_violations_steps"] = []
metrics[LB_TRACK_KEY]["lb_last_threshold"] = []

# Schritt 0 (vor erster Flipoperation)
for name, S in CUTS.items():
    s0, _, cut_edges0 = Calc_updated.cut_metrics(current_G, S, d)
    c0 = len(cut_edges0)
    metrics[name]["strain"].append(s0)
    metrics[name]["exp_exact"].append(np.nan)       # Expected ab Schritt 1
    metrics[name]["exp_alt"].append(np.nan)         # Alt optional/periode
    metrics[name]["exp_alt_last_step"].append(None)
    metrics[name]["cut_size"].append(c0)

    if name == LB_TRACK_KEY:
        metrics[name]["exp_cut_size_lb"].append(np.nan)    # in Schritt 0 noch keine "vor dem Flip"-Schwelle
        metrics[name]["lb_last_threshold"].append(np.nan)
    else:
        metrics[name]["exp_cut_size_lb"].append(np.nan)

    metrics[name]["under_lb_flag"].append(False)

amount_flip_operations = 0

# Hauptschleife: Schritte 1..upper_bound
for t in range(1, upper_bound + 1):
    amount_flip_operations += 1

    # --- VOR DEM FLIP: nur für LB-Cut CutSize/Kanten merken + Schwelle für diesen Schritt
    s_prev_LB, _, cut_edges_prev_LB = Calc_updated.cut_metrics(current_G, CUTS[LB_TRACK_KEY], d)
    c_prev_LB = len(cut_edges_prev_LB)
    pre_cut_edges_LB = set(cut_edges_prev_LB)

    # Per-Step-Schwelle für LB-Cut (mit + δ):
    lb_threshold_t = c_prev_LB + 2.0 * (1.0 - delta_cut_size * c_prev_LB)
    metrics[LB_TRACK_KEY]["exp_cut_size_lb"].append(lb_threshold_t)
    metrics[LB_TRACK_KEY]["lb_last_threshold"].append(lb_threshold_t)

    # --- Für EXPECTED_exact(t) brauchen wir ACTUAL(t-1) auf dem Vor-Graphen
    prev_strain_for_exact = {}
    if compute_exact and (t % exact_period == 0):
        for name, S in CUTS.items():
            prev_strain_for_exact[name] = Calc_updated.cut_metrics(current_G, S, d)[0]

    # --- FLIP durchführen (mit hub-edge)
    current_G, removed, added, hub = Graph.flip_operation(current_G, number_of_cliques, size_of_cliques)
    changed = (removed is not None and added is not None)
    hub_edges.append(hub)

    # Visualisierung: Snapshot speichern
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

    # --- NACH DEM FLIP: Metriken + LB-Prüfung
    for name, S in CUTS.items():
        if not changed:
            # Keine Änderungen: Werte übernehmen / Flags anhängen
            metrics[name]["strain"].append(metrics[name]["strain"][t-1])
            metrics[name]["exp_exact"].append(metrics[name]["exp_exact"][t-1])
            metrics[name]["exp_alt"].append(metrics[name]["exp_alt"][t-1])
            metrics[name]["exp_alt_last_step"].append(metrics[name]["exp_alt_last_step"][t-1])
            metrics[name]["cut_size"].append(metrics[name]["cut_size"][t-1])
            # Für Nicht-LB-Cuts: exp_cut_size_lb ggf. auf NaN auffüllen
            if name != LB_TRACK_KEY:
                metrics[name]["exp_cut_size_lb"].append(np.nan)
            # LB-Flag (dieser Schritt)
            metrics[name]["under_lb_flag"].append(False)
            continue

        # Fälligkeitslogik
        need_actual_t = (t % actual_period == 0)
        need_exact_t  = (compute_exact and (t % exact_period == 0))
        need_alt_t    = (compute_alt and (t % alt_period == 0))

        # ACTUAL(t) & CUTSIZE(t) (CutSize brauchen wir immer fortlaufend)
        strain_t, _, cut_edges_t = Calc_updated.cut_metrics(current_G, S, d)
        cut_size_t = len(cut_edges_t)

        # ACTUAL(t)
        if need_actual_t or need_exact_t:
            metrics[name]["strain"].append(strain_t)
            metrics[name]["cut_size"].append(cut_size_t)
        else:
            metrics[name]["strain"].append(metrics[name]["strain"][t-1])
            metrics[name]["cut_size"].append(cut_size_t)

        # EXPECTED_exact(t): nutzt ACTUAL(t-1)
        if need_exact_t:
            exp_exact_t = Calc_updated.expected_cut_strain_exact(
                current_G, S, d, prev_strain_for_exact[name]
            )
            metrics[name]["exp_exact"].append(exp_exact_t)
        else:
            metrics[name]["exp_exact"].append(metrics[name]["exp_exact"][t-1])

        # EXPECTED_alt(t): optional + periodisch
        if need_alt_t:
            exp_alt_t = Calc_updated.calculate_expected_cut_strain_alternative(current_G, S, d, strain_t)
            metrics[name]["exp_alt"].append(exp_alt_t)
            metrics[name]["exp_alt_last_step"].append(t)
        else:
            metrics[name]["exp_alt"].append(metrics[name]["exp_alt"][t-1])
            metrics[name]["exp_alt_last_step"].append(metrics[name]["exp_alt_last_step"][t-1])

        # exp_cut_size_lb fortschreiben (NaN) für Nicht-LB-Cuts
        if name != LB_TRACK_KEY:
            metrics[name]["exp_cut_size_lb"].append(np.nan)

        # --- LB-Prüfung nur für LB-Cut, nur wenn initial eligible und Hub-Edge kreuzte
        if name == LB_TRACK_KEY:
            under_lb = False
            if LB_ELIGIBLE and (hub is not None):
                a, b = hub if isinstance(hub, tuple) else tuple(hub)
                # Hub-Edge vor dem Flip im Cut?
                hub_in_cut = ((a, b) in pre_cut_edges_LB) or ((b, a) in pre_cut_edges_LB)
                if hub_in_cut:
                    # Schwelle basiert auf c_prev_LB (vor dem Flip) mit "+"-Formel:
                    threshold = metrics[LB_TRACK_KEY]["lb_last_threshold"][-1]
                    if np.isfinite(threshold) and (cut_size_t < threshold):
                        under_lb = True
                        metrics[LB_TRACK_KEY]["lb_violations_steps"].append(t)
            metrics[name]["under_lb_flag"].append(under_lb)
        else:
            metrics[name]["under_lb_flag"].append(False)

    # Fortschritt alle ~1% (mind. alle 1 Versuche)
    report_every = max(1, upper_bound // 100)
    if amount_flip_operations % report_every == 0:
        percentage = (amount_flip_operations * 100.0) / upper_bound
        print(f"{percentage:.0f} %  |  attempts: {amount_flip_operations}")

# Schritte für die Plots (aus der Metrik-Länge ablesen)
steps = list(range(len(next(iter(metrics.values()))["strain"])))  # 0..T
total_steps = len(steps) - 1

# =========================
# Plot-Layout (mit zweigeteiltem Info-Panel bei Visualisierung)
# =========================
fig = plt.figure(figsize=(14, 9))

if visualize_graph:
    # 2 Spalten: links Strain-Charts; rechts Info + (Graph ODER Cut-Size-Charts)
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.50, 1.0, 1.0, 1.0],
        left=0.05, right=0.98, bottom=0.10, top=0.95,
        wspace=0.25, hspace=0.25
    )
    # Links: drei Strain-Charts
    ax_cut1  = fig.add_subplot(gs[1, 0])
    ax_cut2  = fig.add_subplot(gs[2, 0], sharex=ax_cut1)
    ax_cut3  = fig.add_subplot(gs[3, 0], sharex=ax_cut1)

    # Info-Panel links & rechts (jeweils obere Reihe)
    ax_info_left  = fig.add_subplot(gs[0, 0])
    ax_info_right = fig.add_subplot(gs[0, 1])

    # Rechte untere Fläche: Graph ODER Cut-Size-Charts
    ax_graph = fig.add_subplot(gs[1:, 1])

    # Cut-Size-Unterfenster (initial versteckt)
    gs_sizes = gs[1:, 1].subgridspec(3, 1, hspace=0.25)
    ax_size1 = fig.add_subplot(gs_sizes[0, 0], sharex=ax_cut1)
    ax_size2 = fig.add_subplot(gs_sizes[1, 0], sharex=ax_cut1)
    ax_size3 = fig.add_subplot(gs_sizes[2, 0], sharex=ax_cut1)
    for ax in (ax_size1, ax_size2, ax_size3):
        ax.set_visible(False)

else:
    # Keine Graph-Visualisierung: Info über beide Spalten
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.50, 1.0, 1.0, 1.0],
        left=0.05, right=0.98, bottom=0.10, top=0.95,
        wspace=0.25, hspace=0.25
    )
    ax_info = fig.add_subplot(gs[0, :])
    ax_cut1  = fig.add_subplot(gs[1, 0])
    ax_cut2  = fig.add_subplot(gs[2, 0], sharex=ax_cut1)
    ax_cut3  = fig.add_subplot(gs[3, 0], sharex=ax_cut1)
    ax_size1 = fig.add_subplot(gs[1, 1], sharex=ax_cut1)
    ax_size2 = fig.add_subplot(gs[2, 1], sharex=ax_cut1)
    ax_size3 = fig.add_subplot(gs[3, 1], sharex=ax_cut1)
    ax_graph = None

    # Im No-Graph-Fall zeigen beide Variablen auf dieselbe Info-Achse
    ax_info_left  = ax_info
    ax_info_right = ax_info

# Beide Info-Achsen „unsichtbar“
ax_info_left.set_axis_off()
ax_info_right.set_axis_off()

# =========================
# Plots & Legende
# =========================
def plot_cut_metrics(ax, name, show_xlabel=False):
    y_strain = metrics[name]["strain"]
    y_exact  = metrics[name]["exp_exact"]

    lines = []
    l1, = ax.plot(steps, y_strain, label="Actual Cut Strain")
    lines.append(l1)

    if compute_exact:
        y_exact_plot = np.array(y_exact, dtype=float)
        if len(y_exact_plot) > 0:
            y_exact_plot[0] = np.nan
        l2, = ax.plot(steps, y_exact_plot, label="Expected Cut Strain (Giakkoupis)")
        lines.append(l2)

    ax.set_title(CUT_TITLES.get(name, name))
    ax.grid(True, alpha=0.3)
    if len(steps) > 1:
        ax.set_xlim(0, len(steps) - 1)
    else:
        ax.set_xlim(-0.5, 0.5)

    if show_xlabel:
        ax.set_xlabel("Flip-Index")
    return tuple(lines)

def plot_cut_sizes(ax, name, show_xlabel=False):
    y_size = metrics[name]["cut_size"]
    ax.plot(steps, y_size)
    ax.set_title(CUT_TITLES.get(name, name) + " – Cut Size")
    ax.grid(True, alpha=0.3)
    if len(steps) > 1:
        ax.set_xlim(0, len(steps) - 1)
    else:
        ax.set_xlim(-0.5, 0.5)
    if show_xlabel:
        ax.set_xlabel("Flip-Index")
    ax.set_ylabel("Cut Size")

# Strain Plots links
lines_top = plot_cut_metrics(ax_cut1, "biggest")
_ = plot_cut_metrics(ax_cut2, f"block_{k_block}")
_ = plot_cut_metrics(ax_cut3, "every_second", show_xlabel=True)
ax_cut2.set_ylabel("Wert")

# Size Plots (rechts unten – sichtbar nur ohne Graph; mit Graph via View)
plot_cut_sizes(ax_size1, "biggest")
plot_cut_sizes(ax_size2, f"block_{k_block}")
plot_cut_sizes(ax_size3, "every_second", show_xlabel=True)
if visualize_graph:
    for ax in (ax_size1, ax_size2, ax_size3):
        ax.set_visible(False)

# Eine Legende (oben), dynamisch nach Anzahl Linien
legend_handles = list(lines_top)
fig.legend(
    handles=legend_handles,
    labels=[ln.get_label() for ln in legend_handles],
    loc="upper center", bbox_to_anchor=(0.5, 0.995),
    ncol=len(legend_handles), frameon=False
)

# Cursor-Linien (Strain)
cursor1 = ax_cut1.axvline(x=0, linestyle="--", color="grey", linewidth=1)
cursor2 = ax_cut2.axvline(x=0, linestyle="--", color="grey", linewidth=1)
cursor3 = ax_cut3.axvline(x=0, linestyle="--", color="grey", linewidth=1)

# Cursor-Linien (Size)
cursor_s1 = ax_size1.axvline(x=0, linestyle="--", color="grey", linewidth=1)
cursor_s2 = ax_size2.axvline(x=0, linestyle="--", color="grey", linewidth=1)
cursor_s3 = ax_size3.axvline(x=0, linestyle="--", color="grey", linewidth=1)

# =========================
# Info-Panel + Controls (rechts andocken)
# =========================
# BBox der rechten Info-Achse für Dropdown/View-Controls
info_bbox = ax_info_right.get_position()

# --- Schlankes Cut-Auswahl-Widget (rechts außen)
label_by_key = {k: CUT_TITLES.get(k, k) for k in CUTS.keys()}
key_by_label = {v: k for k, v in label_by_key.items()}
labels = [label_by_key[k] for k in CUTS.keys()]
selected_cut_key = [list(CUTS.keys())[0]]  # initial: "biggest"

drop_w = 0.16 * info_bbox.width
drop_h = 0.50 * info_bbox.height
drop_ax = fig.add_axes([
    info_bbox.x0 + info_bbox.width - drop_w - 0.004,
    info_bbox.y0 + 0.44 * info_bbox.height,
    drop_w,
    drop_h
])
drop_ax.set_facecolor('none')
for spine in drop_ax.spines.values():
    spine.set_visible(False)
drop_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

def on_cut_changed(label):
    if label in key_by_label:
        selected_cut_key[0] = key_by_label[label]
    else:
        selected_cut_key[0] = key_by_label.get(str(label), selected_cut_key[0])
    update_all(current_step[0])

def _bind_dropdown(ax, labels, initial_label):
    global dropdown
    try:
        from matplotlib.widgets import Dropdown as MplDropdown
        dropdown = MplDropdown(ax, label='', options=labels, value=initial_label)
        dropdown.on_changed(on_cut_changed)
        try:
            for child in ax.get_children():
                if hasattr(child, 'set_fontsize'):
                    child.set_fontsize(8)
        except Exception:
            pass
        return "dropdown"
    except Exception:
        from matplotlib.widgets import RadioButtons
        ax.clear()
        dropdown = RadioButtons(ax, labels, active=labels.index(initial_label))
        dropdown.on_clicked(on_cut_changed)
        try:
            for lbl in dropdown.labels:
                lbl.set_fontsize(8)
            for circ in dropdown.circles:
                circ.set_radius(0.035)
        except Exception:
            pass
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        return "radio"

_ui_type = _bind_dropdown(drop_ax, labels, label_by_key[selected_cut_key[0]])

# --- View-Selector (schmal, unterhalb des Cut-Dropdowns)
if visualize_graph:
    view_w = drop_w
    view_h = 0.30 * info_bbox.height
    view_ax = fig.add_axes([
        info_bbox.x0 + info_bbox.width - view_w - 0.004,
        info_bbox.y0 + 0.08 * info_bbox.height,
        view_w,
        view_h
    ])
    view_ax.set_facecolor('none')
    for spine in view_ax.spines.values():
        spine.set_visible(False)
    view_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    def _set_view_from_label(label):
        lbl = str(label).strip().lower()
        mode = "Graph" if "graph" in lbl else "Cut Size"
        set_view(mode)
        update_all(current_step[0])

    def _bind_view_selector(ax, initial_value="Graph"):
        try:
            from matplotlib.widgets import Dropdown as MplDropdown
            view_selector = MplDropdown(ax, label='', options=["Graph", "Cut Size"], value=initial_value)
            view_selector.on_changed(_set_view_from_label)
            try:
                for child in ax.get_children():
                    if hasattr(child, 'set_fontsize'):
                        child.set_fontsize(10)
            except Exception:
                pass
            return ("dropdown", view_selector)
        except Exception:
            from matplotlib.widgets import RadioButtons
            ax.clear()
            view_selector = RadioButtons(ax, ("Graph", "Cut Size"),
                                         active=0 if initial_value == "Graph" else 1)
            view_selector.on_clicked(_set_view_from_label)
            try:
                for lbl in view_selector.labels:
                    lbl.set_fontsize(10)
                for circ in view_selector.circles:
                    circ.set_radius(0.045)
            except Exception:
                pass
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            return ("radio", view_selector)

    _view_ui_type, _view_ui = _bind_view_selector(view_ax, initial_value="Graph")

# Zusätzlich unten: TextBox "Go to step"
axbox = plt.axes([0.08, 0.02, 0.18, 0.05])
step_box = TextBox(axbox, 'Go to step: ', initial='0')

# =========================
# Zeichenfunktionen
# =========================
def draw_graph_at_step(i: int):
    if not visualize_graph or ax_graph is None or not ax_graph.get_visible():
        return
    ax_graph.clear()
    G = graphs[i]
    removed_edges, added_edges = flip_info[i] if i < len(flip_info) else (set(), set())

    # Knotenfarben: ausgewählten Cut gelb markieren
    S = CUTS[selected_cut_key[0]]
    node_cols = ["yellow" if (u in S) else "lightblue" for u in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_cols,
                           node_size=80 if len(G) > 150 else 120)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color="black", width=1.2)

    # Hinzugefügte Kanten in diesem Schritt: rot
    added_list = [(u, v) for (u, v) in added_edges if G.has_edge(u, v)]
    if added_list:
        nx.draw_networkx_edges(G, pos, edgelist=added_list, ax=ax_graph,
                               edge_color="red", width=2.0)

    # Entfernte Kanten gestrichelt blau
    for (u, v) in removed_edges:
        if u in pos and v in pos:
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            ax_graph.plot(x, y, linestyle="--", color="blue", linewidth=1.6, alpha=0.9, zorder=0)

    if draw_labels:
        labels_nodes = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels_nodes, ax=ax_graph, font_size=6)

    ax_graph.set_title(f"Ring of Cliques – Schritt {i}/{total_steps}")
    ax_graph.set_axis_off()

def format_edges(edges_set):
    if not edges_set:
        return "-"
    try:
        seq = sorted(list(edges_set))
    except TypeError:
        seq = list(edges_set)
    return ", ".join([f"({u},{v})" for (u, v) in seq])

def update_info_panel(i: int):
    # Beide Info-Achsen leeren
    ax_info_left.clear();  ax_info_left.set_axis_off()
    ax_info_right.clear(); ax_info_right.set_axis_off()

    removed_edges, added_edges = flip_info[i] if i < len(flip_info) else (set(), set())
    k = selected_cut_key[0]
    title = CUT_TITLES.get(k, k)

    val_strain = metrics[k]["strain"][i]
    val_csize  = metrics[k]["cut_size"][i]

    # Expected (exact): Anzeige abhängig von compute_exact
    if not compute_exact:
        val_exact_str = "N/A"
        exact_suffix = "(deaktiviert)"
    else:
        if i == 0 or not np.isfinite(metrics[k]["exp_exact"][i]):
            val_exact_str = "N/A"
            exact_suffix = f"(Interval: {exact_period})"
        else:
            val_exact_str = f"{metrics[k]['exp_exact'][i]:.4f}"
            exact_suffix = f"(Interval: {exact_period})"

    # Alternative
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
            alt_suffix = f"(zuletzt berechnet bei Schritt {last_alt_step}; Period: {alt_period})"

    lines = [
        f"Schritt {i}/{total_steps}   |   Ausgewählter Cut: {title}",
        f"Actual Cut Strain: {val_strain:.4f}  (Interval: {actual_period})",
        f"Expected Cut Strain (Giakkoupis): {val_exact_str} {exact_suffix}",
        f"Expected Cut Strain (alternative): {val_alt_str} {alt_suffix}",
        "",
        f"Cut Size c_G'(S): {val_csize}",
    ]

    # --- LB-Infos NUR anzeigen, wenn der ausgewählte Cut der smallest cut ist
    if selected_cut_key[0] == LB_TRACK_KEY:
        # aktuelle per-step Schwelle (vor dem Flip berechnet; für i=0 meist NaN)
        try:
            lb_step_val = metrics[LB_TRACK_KEY]["exp_cut_size_lb"][i]
        except Exception:
            lb_step_val = np.nan

        viol_steps = metrics[LB_TRACK_KEY].get("lb_violations_steps", [])
        viol_count = len(viol_steps)
        viol_list_str = ", ".join(map(str, viol_steps)) if viol_steps else "-"

        lines += [
            f"LB check active?  {'YES' if LB_ELIGIBLE else 'no'}  (initial cut size = {c0_init}, d = {d})",
            f"LB per-step threshold (only if hub-edge crosses): c_prev + 2*(1 + δ*c_prev), δ={delta_cut_size}",
            f"Threshold this step (smallest cut): {('N/A' if not np.isfinite(lb_step_val) else f'{lb_step_val:.2f}')}",
            f"LB violations so far (smallest cut): {viol_count}  |  Steps: {viol_list_str}",
            "",
        ]

    # Flip-Edges (immer)
    lines += [
        f"Removed: {format_edges(removed_edges)}",
        f"Added:   {format_edges(added_edges)}",
    ]

    # --- Zweispaltiges Layout: Links erste 7, rechts der Rest ---
    split_at = 7
    left_lines  = lines[:split_at]
    right_lines = lines[split_at:]

    # Linke Spalte
    ax_info_left.text(0.01, 0.95, "\n".join(left_lines),
                      va="top", ha="left", fontsize=10)

    # Rechte Spalte
    if right_lines:
        ax_info_right.text(0.02, 0.95, "\n".join(right_lines),
                           va="top", ha="left", fontsize=10)

# =========================
# Navigation + View
# =========================
current_step = [0]
view_mode = ["Graph" if visualize_graph else "Cut Size"]  # Startmodus

def set_view(mode: str):
    view_mode[0] = mode
    if ax_graph is not None:
        if mode == "Graph":
            ax_graph.set_visible(True)
            for ax in (ax_size1, ax_size2, ax_size3):
                ax.set_visible(False)
        else:
            ax_graph.set_visible(False)
            for ax in (ax_size1, ax_size2, ax_size3):
                ax.set_visible(True)
    fig.canvas.draw_idle()

def update_all(i: int):
    # Cursor updaten (Strain)
    for ln in (cursor1, cursor2, cursor3):
        ln.set_xdata([i, i])
    # Cursor updaten (Size)
    for ln in (cursor_s1, cursor_s2, cursor_s3):
        ln.set_xdata([i, i])
    # Info-Panel + (optional) Graph/Cut-Size neu zeichnen
    update_info_panel(i)
    if ax_graph is not None and view_mode[0] == "Graph":
        draw_graph_at_step(i)
    fig.canvas.draw_idle()

def on_next(event):
    current_step[0] = (current_step[0] + 1) % len(steps)
    update_all(current_step[0])

def on_prev(event):
    current_step[0] = (current_step[0] - 1) % len(steps)
    update_all(current_step[0])

def on_submit_step(text):
    try:
        x = int(text)
    except Exception:
        return
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

# TextBox-Callback
step_box.on_submit(on_submit_step)

# Initial zeichnen
update_all(current_step[0])
plt.show()
