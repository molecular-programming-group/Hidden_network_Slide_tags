import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
from matplotlib.collections import LineCollection

# Helper function to ensure directory creation
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Filtering beads based on nUMI sum thresholds
def nUMI_sum_filtering(edge_list, nUMI_thresholds, out_dir, include_N_beads=False):
    edge_list_abundant_beads_cut = edge_list.set_index("bead_bc")
    filename_numi_sum = os.path.join(out_dir, f"edge_list_filtered_by_nUMI_beadsum_thresholds_{nUMI_thresholds[0]}_{nUMI_thresholds[1]}.csv")

    if os.path.isfile(filename_numi_sum): # if this file is already made, load it
        return pd.read_csv(filename_numi_sum)

    filtering_df = edge_list_abundant_beads_cut.groupby("bead_bc").sum()
    beads_to_cut = [
        bead for bead, nUMI_sum_collapsed in filtering_df.iterrows()
        if "N" in bead and not include_N_beads or nUMI_sum_collapsed['nUMI'] > nUMI_thresholds[1] or nUMI_sum_collapsed['nUMI'] < nUMI_thresholds[0]
    ]

    edge_list_abundant_beads_cut = edge_list_abundant_beads_cut[~edge_list_abundant_beads_cut.index.isin(beads_to_cut)]
    edge_list_filtered_by_nUMI_sum = edge_list_abundant_beads_cut.reset_index()
    edge_list_filtered_by_nUMI_sum.to_csv(filename_numi_sum, index=False)

    return edge_list_filtered_by_nUMI_sum

# Filtering beads based on number of connections
def n_connections_filtering(edge_list, n_connections_thresholds, out_dir):
    filename_n_connections = os.path.join(out_dir, f"edge_list_filtered_by_bed_n_connections_thresholds_{n_connections_thresholds[0]}-{n_connections_thresholds[1]}.csv")

    if os.path.isfile(filename_n_connections):
        return pd.read_csv(filename_n_connections)

    all_beads_post_sum_filter = np.unique(edge_list["bead_bc"], return_counts=True)
    beads_to_cut = [
        bead for bead, count in zip(all_beads_post_sum_filter[0], all_beads_post_sum_filter[1])
        if count < n_connections_thresholds[0] #check if its lower than the lower threshold 
        or count > n_connections_thresholds[1] #or higher than the upper threshold
    ]

    edge_list_filtered = edge_list[~edge_list["bead_bc"].isin(beads_to_cut)]
    edge_list_filtered.to_csv(filename_n_connections, index=False)

    return edge_list_filtered

# Filtering edges based on per-edge weight threshold
def per_edge_weight_filtering(edge_list, per_edge_weight_threshold, out_dir):
    filename_per_edge_weight = os.path.join(out_dir, f"edge_list_filtered_by_per_edge_weight_{per_edge_weight_threshold}.csv")

    if os.path.isfile(filename_per_edge_weight):
        return pd.read_csv(filename_per_edge_weight)

    edge_list_filtered = edge_list[edge_list["nUMI"] >= per_edge_weight_threshold].reset_index(drop=True)
    edge_list_filtered.to_csv(filename_per_edge_weight, index=False)

    return edge_list_filtered

# Generating bead-cell weight matrix
def generate_bead_cell_weight_matrix(edge_list, out_dir):
    filename_cell_bead_mtx = os.path.join(out_dir, "bead-cell_weight_matrix.csv")

    if os.path.isfile(filename_cell_bead_mtx):
        return pd.read_csv(filename_cell_bead_mtx, index_col=0)

    unique_bead_bcs = edge_list["bead_bc"].unique()
    unique_cell_bcs = edge_list["cell_bc_10x"].unique()
    weights_bead_to_cells = pd.DataFrame(0, index=unique_bead_bcs, columns=unique_cell_bcs)

    for _, row in edge_list.iterrows():
        weights_bead_to_cells.at[row['bead_bc'], row['cell_bc_10x']] = row['nUMI']

    weights_bead_to_cells.to_csv(filename_cell_bead_mtx)

    return weights_bead_to_cells

# Converting bead-cell weight matrix to cell-cell weight matrix
def convert_bead_cell_to_cell_cell(weights_bead_to_cells, out_dir):
    filename_cell_cell = os.path.join(out_dir, "cell-cell_weight_matrix.csv")
    filename_cell_cell_degree = os.path.join(out_dir, "cell-cell_weight_degree_matrix.csv")

    if os.path.isfile(filename_cell_cell) and os.path.isfile(filename_cell_cell_degree):
        weights_cells_to_cells = pd.read_csv(filename_cell_cell, index_col=0)
        weights_n_beads_cells_to_cells = pd.read_csv(filename_cell_cell_degree, index_col=0)
        return weights_cells_to_cells, weights_n_beads_cells_to_cells

    unique_cell_bcs = weights_bead_to_cells.columns
    weights_cells_to_cells = pd.DataFrame(0, index=unique_cell_bcs, columns=unique_cell_bcs)
    weights_n_beads_cells_to_cells = pd.DataFrame(0, index=unique_cell_bcs, columns=unique_cell_bcs)

    for bead, row in weights_bead_to_cells.iterrows():
        connected_cells = row[row > 0].index
        for i, cell1 in enumerate(connected_cells):
            for cell2 in connected_cells[i+1:]:
                weights_cells_to_cells.at[cell1, cell2] += row[cell1]
                weights_n_beads_cells_to_cells.at[cell1, cell2] += 1

    weights_cells_to_cells.to_csv(filename_cell_cell)
    weights_n_beads_cells_to_cells.to_csv(filename_cell_cell_degree)

    return weights_cells_to_cells, weights_n_beads_cells_to_cells

# Plotting true positions and edges
def plotting_true_positions_and_edges(nucleus_coordinates, out_dir, weights_cells_to_cells, weights_n_beads_cells_to_cells, args):
    n_beads_threshold = args["n_beads_threshold"]
    spatial_bc = nucleus_coordinates["NAME"][1:]
    nucleus_coordinates_speed_test = nucleus_coordinates.set_index("NAME")

    edges = []
    all_distances, all_umi_weights, all_bead_weights = [], [], []
    cells_to_plot_x, cells_to_plot_y = [], []
    for idx, (cell, row) in enumerate(weights_cells_to_cells.iterrows()):

        cell_spatial = cell + "-1" if "Mouse_Embryo" in out_dir else cell
        x_base_cell = float(nucleus_coordinates_speed_test.at[cell_spatial, "X"])
        y_base_cell = float(nucleus_coordinates_speed_test.at[cell_spatial, "Y"])

        if cell_spatial not in spatial_bc.values:
            print("huh")
            continue

        for connected_cell in row[row > 0].index:

            connected_cell_spatial = connected_cell + "-1" if "Mouse_Embryo" in out_dir else connected_cell
            if connected_cell_spatial not in spatial_bc.values:
                continue

            x_connected_cell = float(nucleus_coordinates_speed_test.at[connected_cell_spatial, "X"])
            y_connected_cell = float(nucleus_coordinates_speed_test.at[connected_cell_spatial, "Y"])

            cells_to_plot_x.extend([x_base_cell, x_connected_cell])
            cells_to_plot_y.extend([y_base_cell, y_connected_cell])
            edges.append([[x_base_cell, y_base_cell], [x_connected_cell, y_connected_cell]])

            distance = np.sqrt((x_base_cell - x_connected_cell) ** 2 + (y_base_cell - y_connected_cell) ** 2)
            umi_weight = row[connected_cell]
            bead_weight = weights_n_beads_cells_to_cells.at[cell, connected_cell]

            all_distances.append(distance)
            all_umi_weights.append(umi_weight)
            all_bead_weights.append(bead_weight)

    if args["generate_plots"] == True:
        fig, axes = plt.subplots(2, 3, figsize=(12, 9))
        ax_umi_weight_edges, ax_bead_weight_edges, ax_filtered_n_beads = axes[0]
        ax_threshold_1, ax_threshold_2, ax_threshold_3 = axes[1]

        for ax, threshold in zip([ax_filtered_n_beads, ax_threshold_1, ax_threshold_2, ax_threshold_3], [n_beads_threshold, n_beads_threshold + 1, n_beads_threshold + 3, n_beads_threshold + 5]):
            filtered_edges = [edges[i] for i in range(len(all_bead_weights)) if all_bead_weights[i] >= threshold]
            lines = LineCollection(filtered_edges, color="k", linewidth=0.5)
            ax.add_collection(lines)
            ax.scatter(nucleus_coordinates_speed_test[1:]["X"].astype("float"), nucleus_coordinates_speed_test[1:]["Y"].astype("float"), s=5, c="lightgray")
            ax.set_box_aspect(1)
            ax.set_title(f"nBeads weight cutoff {threshold}")

        ax_umi_weight_edges.hist(all_umi_weights, bins=np.arange(0.5, np.max(all_umi_weights) + 1.5, 1), align='mid')
        ax_umi_weight_edges.set_yscale("log")
        ax_umi_weight_edges.set_xscale("log")
        ax_umi_weight_edges.set_title("distribution of UMI weights")
        ax_umi_weight_edges.set_xlabel("UMIs per cell-cell edge")
        ax_umi_weight_edges.set_ylabel("Count")
        ax_bead_weight_edges.hist(all_bead_weights, bins=np.arange(0.5, np.max(all_umi_weights) + 1.5, 1), align='mid')
        ax_bead_weight_edges.set_yscale("log")
        ax_bead_weight_edges.set_xscale("log")
        ax_bead_weight_edges.set_title("distribution of bead weights")
        ax_bead_weight_edges.set_xlabel("Beads per cell-cell edge")
        ax_bead_weight_edges.set_ylabel("Count")
        fig.savefig(os.path.join(out_dir, "Figures", f"edges_drawn_out_and_cutoff_{n_beads_threshold}.png"), format="png", dpi=1200)
        plt.close(fig)

    return all_distances, all_bead_weights, all_umi_weights

# Main function to process all files
def main(args):

    folder_name = args["Sample"]
    entries = os.listdir(folder_name)
    spatial_file = next(entry for entry in entries if "spatial.csv" in entry)
    nucleus_coordinates = pd.read_csv(os.path.join(folder_name, spatial_file))

    date = datetime.now()
    date = date.strftime("%Y%m%d_t=%H.%M")
    date = "20240701_t=17.15"
    Edge_files = args["processed_edge_files"]
    nUMI_thresholds_list = args["nUMI_sum_per_bead_thresholds"]
    n_connections_thresholds_list = args["n_connected_cells_thresholds"]
    per_edge_weight_threshold_list = args["per_edge_nUMI_thresholds"]

    include_N_beads = args["include_N_beads"]

    for unfiltered_edges_file in Edge_files:
        edge_list = pd.read_csv(unfiltered_edges_file, sep=",", index_col="Unnamed: 0") if "spatial" in unfiltered_edges_file else pd.read_csv(unfiltered_edges_file, sep=",")
        out_dir_base = f"{folder_name}/{date}_spatial"

        if include_N_beads:
            out_dir_base += "_with_N"

        for per_edge_weight_threshold in per_edge_weight_threshold_list:
            args["n_beads_threshold"] = 1 if per_edge_weight_threshold != 1 else args["n_beads_threshold"]
            for n_connections_thresholds in n_connections_thresholds_list:
                for nUMI_thresholds in nUMI_thresholds_list:
                    out_dir = f"{out_dir_base}_bead_sum{nUMI_thresholds[0]}-{nUMI_thresholds[1]}_edge_{per_edge_weight_threshold}_n_connections_{n_connections_thresholds[0]}-{n_connections_thresholds[1]}"
                    ensure_directory(out_dir)
                    print(f"Filtering based on bead UMI sums with thresholds {nUMI_thresholds}")
                    edge_list_filtered_by_nUMI_sum = nUMI_sum_filtering(edge_list, nUMI_thresholds, out_dir, include_N_beads)
                    print(f"Filtering based on number of cells connected per bead with thresholds {n_connections_thresholds}")
                    edge_list_bead_n_connections_filtered = n_connections_filtering(edge_list_filtered_by_nUMI_sum, n_connections_thresholds, out_dir)
                    print(f"Filtering edges by number of UMIs per individual edge: {per_edge_weight_threshold}")
                    edge_list_filtered_per_edge_w = per_edge_weight_filtering(edge_list_bead_n_connections_filtered, per_edge_weight_threshold, out_dir)
                    print("Converting to bead-cell biadjacency matrix")
                    weights_bead_to_cells = generate_bead_cell_weight_matrix(edge_list_filtered_per_edge_w, out_dir)
                    print("Converting to cell-cell adjacency matrix")
                    weights_cells_to_cells, weights_n_beads_cells_to_cells = convert_bead_cell_to_cell_cell(weights_bead_to_cells, out_dir)
                    print(f"Plotting based on bead threshold {args["n_beads_threshold"]}")
                    plotting_true_positions_and_edges(nucleus_coordinates, out_dir, weights_cells_to_cells, weights_n_beads_cells_to_cells, args)
                    plt.close("all")

# if __name__ == "__main__":
args = {
    "Sample": "Mouse_Embryo", 
    "processed_edge_files": [
        "SRR11_edge_list_sequences_only_spatial.csv"
        ], 
    "nUMI_sum_per_bead_thresholds": [[1, 256]], 
    "n_connected_cells_thresholds": [[1, 256]], #having these be the same as the filtering thresholds means no filtering is done in practiae except the first step
    "per_edge_nUMI_thresholds": [1], 
    "n_beads_threshold": 4,
    "generate_plots": True,
    "include_N_beads": False
    }

main(args)
