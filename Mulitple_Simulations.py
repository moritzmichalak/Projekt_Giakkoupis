import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.widgets import Button
import Graph
import Calc_updated


# User input

print("Choose number of nodes:")
n = int(input())
# Get all possible degrees to limit possiblity of invalid input
possible_d = Calc_updated.calculate_possible_d(n)
print("Choose degree from ", possible_d)
chosen_d = int(input())
for i in possible_d:
    if chosen_d == i:
        d = chosen_d
number_of_cliques = int(n / (d+1))
size_of_cliques = d+1
print("How many simulations do you want to run:")
num_simulations = int(input())
print("You chose a graph with ", number_of_cliques,
      " cliques, each with a size of", size_of_cliques, " nodes.")
# Option: Run till upper bound of Giakkoupis or stop after convergance
print("Run all flips? y or n")
flips = input()
if flips == "y":
    all_flips = True
else:
    all_flips = False

# Create Graph with parameters from user
G, pos = Graph.create_ring_of_cliques(number_of_cliques, size_of_cliques)

# These global settings determine how often metrics get calculated to save running time and costs.
MEASURING_RATE_HIGH = math.ceil(n/20)
MEASURING_RATE_LOW = math.ceil(n/10)

# Global maximum flips (optional)
max_flips = 5000000000000

# Global setting: How often does the threshold need to be hit to make sure the graph is an expander.
REQUIRED_HITS = 15
COUNT_CONSECUTIVE = True          # True for consecutive hits
# Maximum number of flips at which the sim stops.
global_stop_at_flips = None

# As seen in Giakkoupis' paper
real_upper_bound = int(n * d * (math.log2(n)) ** 2)
# If global maximum is lower pick that mumber.
upper_bound = min(int(n * d * (math.log2(n)) ** 2), max_flips)

threshold = Calc_updated.recommend_threshold_by_sampling(n, d)  # Threshold
print("Threshold: ", threshold)
print("Upper bound: ", upper_bound)


# This function creates a buffer around the threshold where the graph needs to be for it to be an expander
def compute_band(threshold, rel=0.10, abs_min=1e-3, abs_max=None):
    band = max(abs_min, rel * abs(float(threshold)))
    if abs_max is not None:
        band = min(band, abs_max)
    return band


# 5 percent of the threshold, at least 1e-3
THRESHOLD_BAND = compute_band(threshold, rel=0.05, abs_min=1e-3)

# Arrays for results
all_specs = []
simulations = []  # speichert Daten jeder einzelnen Simulation

# multiple sims
for sim in range(num_simulations):
    nodes = list(G.nodes)
    high_prescision = True  # start with high precision
    print("Simulation", sim + 1, "of", num_simulations)
    current_G = G.copy()
    specvals = []
    flip_info = []

    spec = Calc_updated.spectral_gap_normalized_sparse(
        current_G, d)  # calculate spectral gap
    specvals.append(spec)

    # counter for hits around threshold
    hits_in_band = 0
    flips_done = 0

    # upper bound for this sim
    this_upper = upper_bound if global_stop_at_flips is None else global_stop_at_flips

    while flips_done < this_upper:
        # Flip operation
        current_G, removed, added, _ = Graph.flip_operation(
            current_G, number_of_cliques, size_of_cliques)
        flips_done += 1

        # only measure at measuring rate
        measured_now = False
        if high_prescision:
            if flips_done % MEASURING_RATE_HIGH == 0:
                spec = Calc_updated.spectral_gap_normalized_sparse(
                    current_G, d, tol=5e-3, maxiter=1000)
                measured_now = True
            # First time the graph hits expander levels we reduce the measuring rate
            if spec >= threshold:
                high_prescision = False
        else:
            if flips_done % MEASURING_RATE_LOW == 0:
                spec = Calc_updated.spectral_gap_normalized_sparse(
                    current_G, d, tol=2e-2, maxiter=300)
                measured_now = True

        if measured_now:
            # Make sure if sim needs to run further
            in_band = abs(float(spec) - float(threshold)) <= THRESHOLD_BAND
            if COUNT_CONSECUTIVE:
                hits_in_band = hits_in_band + 1 if in_band else 0
            else:
                if in_band:
                    hits_in_band += 1

            if global_stop_at_flips is None and hits_in_band >= REQUIRED_HITS:
                global_stop_at_flips = min(real_upper_bound, flips_done +
                                           max(math.ceil(upper_bound / 20), 50000))
                print(
                    f"Früher Stopp nach {global_stop_at_flips} Flips festgelegt in Simulation {sim + 1}")

                # nur hier trimmen, weil der Wert jetzt sicher nicht None ist
                _target_spec_len = 1 + global_stop_at_flips
                _target_graphs_len = 1 + global_stop_at_flips
                _target_flips_len = global_stop_at_flips

                for s in simulations:
                    if s.get("specvals") is not None and len(s["specvals"]) > _target_spec_len:
                        del s["specvals"][_target_spec_len:]

                for i in range(len(all_specs)):
                    if len(all_specs[i]) > _target_spec_len:
                        del all_specs[i][_target_spec_len:]

                # set new upper bound for next sims
                this_upper = min(this_upper, global_stop_at_flips)

        # Werte anhängen wie gehabt
        specvals.append(float(spec) if np.isfinite(spec) else np.nan)

        # Progress output
        PROGRESS_EVERY = 20000
        if flips_done % PROGRESS_EVERY == 0:
            pct = (sim * this_upper + flips_done) / \
                (num_simulations * this_upper) * 100
            print(f"{pct:.2f} Prozent, Wert: ",
                  spec, ", Threshhold: ", epsilon, ", hits: ", hits_in_band)

    if global_stop_at_flips is not None:
        _target_spec_len = 1 + global_stop_at_flips
        specvals = specvals[:_target_spec_len]

    simulations.append({
        "specvals": specvals,
    })
    all_specs.append(specvals)


# convert into arrays
# all_specs: list of lists
max_len = max(len(s) for s in all_specs) if all_specs else 0
spec_np = np.full((len(all_specs), max_len), np.nan, dtype=float)
for i, s in enumerate(all_specs):
    spec_np[i, :len(s)] = [v if np.isfinite(v) else np.nan for v in s]

# Durchschnittswerte berechnen
mean_spec = np.nanmean(spec_np, axis=0)
mins = np.nanmin(spec_np, axis=0)
maxs = np.nanmax(spec_np, axis=0)

steps = np.arange(spec_np.shape[1])


# Find sim that hit threshold first
first_hit = []
for sim in simulations:
    s = sim["specvals"]
    idx = next((i for i, v in enumerate(s) if v >= threshold), np.inf)
    first_hit.append(idx)

reachable = [i for i, t in enumerate(first_hit) if np.isfinite(t)]

if reachable:
    best_index = min(
        reachable,
        key=lambda i: (first_hit[i],
                       len(simulations[i]["specvals"]),
                       -simulations[i]["specvals"][-1])
    )
    worst_index = max(
        reachable,
        key=lambda i: (first_hit[i],
                       -len(simulations[i]["specvals"]),
                       simulations[i]["specvals"][-1])
    )
else:
    # Best Case, biggest endresult
    best_index = int(np.argmax([sim["specvals"][-1] for sim in simulations]))
    # Worst Case, smallest endresult
    worst_index = int(np.argmin([sim["specvals"][-1] for sim in simulations]))


# Plot


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

    plt.show()


show_main_plot()
