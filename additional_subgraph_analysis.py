from subgraph_analysis_functions import *
from Utils import *
from matplotlib.collections import LineCollection
from typing import List
from scipy.stats import linregress
import random
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from scipy.spatial import ConvexHull
import os

class gatedSubgraph():
    def __init__(self, config, edgelist_name, predicted_cell_types_file = None):
        
        self.config = config
        self.name = edgelist_name
        self.color_set = self.config.vizualisation_args.colours
        self.colors = self.load_colors()
        self.dimension = self.config.filter_analysis_args.reconstruction_dimension
        self.file_location = self.config.gated_file_location
        self.edgelist_path = f"{self.file_location}/{edgelist_name}"
        self.edgelist = pd.read_csv(self.edgelist_path)
        selected_columns = self.edgelist[[col for col in self.edgelist.columns if col.startswith("distance")]]
        self.n_reconstructions = len(selected_columns.columns)
        self.edgelist["std_distance"] = selected_columns.std(axis = 1)
        self.edgelist["max_diff_distance"] = selected_columns.max(axis=1) - selected_columns.min(axis=1)
        self.sample = self.file_location.split(os.sep)[1]
        try:
            self.base_edges = pd.read_csv(f"Intermediary_files/{self.sample}/all_cells.csv") 
        except:
            self.base_edges = pd.read_csv(f"Intermediary_files/{self.sample}/all_cells_synthetic.csv") 
        self.gating_threshold = re.search(rf"gated_(\d+)", self.name)

        if self.gating_threshold == None:
            self.gating_threshold = "ungated"
            self.gated = False
            if "dbscan" in self.name:
                _, _, full_parameters = self.name.split()[-1].partition("dbscan_")
                self.gating_threshold = full_parameters[:-4]
                self.gated = True
        else:
            all_matches = re.findall(r"_gated_(\d+)", self.name)
            if len(all_matches)==1:
                self.gating_threshold = str(self.gating_threshold.group(1))
            else:
                self.gating_threshold = "+".join(all_matches)
            
            self.gated = True
        self.reconstruction_summary = self.find_reconstruction_summary() 

        self.full_reconstruction_summary = self.find_full_reconstruction()
        self.calculate_per_gt_cell_full_metrics()
        
        # self.max_gt_dist = pdist(self.reconstruction_summary
        # print(self.reconstruction_summary)
        # self.calculate_recon_dbscan_clusters()
    def load_cell_unipartite(self):
        import re

        # Regex pattern
        pattern = r"N=\d+(.*)"
        # Extract match
        match = re.search(pattern, self.name)
        if match:
            result = match.group(1)  # Get everything after 'N=<number>'
        unipartite_path = f"{self.file_location}/unipartite{result}"
        source_nodes = set(self.edgelist["source_bc"].unique())

        if os.path.isfile(unipartite_path):
            print("loading unipartite file")
            self.unipartite_cell = pd.read_csv(unipartite_path)
            print("loaded")
            return
        print("converting to unipartite")

        B = nx.from_pandas_edgelist(self.edgelist, source="source_bc", target="target_bc")
        G = nx.algorithms.bipartite.weighted_projected_graph(B, source_nodes)

        unipartite_df = nx.to_pandas_edgelist(G)
        
        self.unipartite_cell = unipartite_df
        self.unipartite_cell.to_csv(unipartite_path)
        pass
    def save_modified_full_reconstruction_summary(self):
        self.full_reconstruction_summary.to_csv(f"{self.file_location}/modified_reconstruction_summary_gated_{self.gating_threshold}.csv", index = False)

    def load_colors(self):
        
        import importlib
        module = importlib.import_module("cell_colors")
        colors = getattr(module, self.color_set)
        if colors == {}:
            colors = dictRandomcellColors(self.all_ground_truth_points)
        return colors
    
    def plot_edges_gt_estimated_beads(self, ax = None):
        print(self.edgelist)
        gt_positions = self.reconstruction_summary.copy().set_index("bc")
        gt_positions = gt_positions[["gt_x", "gt_y"]]
        import pandas as pd

        # Sample DataFrame names
        df_edges = self.edgelist  # The first DataFrame (edges)
        df_positions = gt_positions  # The second DataFrame (positions)
        df_edges["colors"] = df_edges["source_type"].copy().map(self.colors)

        # Merge to get positions of `source_bc`
        df_merged_sources = df_edges.merge(df_positions, left_on="source_bc", right_index=True, how="left")

        # Compute mean positions for each `target_bc`
        df_avg_positions = df_merged_sources.groupby("target_bc")[["gt_x", "gt_y"]].mean().reset_index()

        # Merge to get `target_bc` positions for each `source_bc`
        df_final = df_edges.merge(df_positions, left_on="source_bc", right_index=True, how="left") \
                        .merge(df_avg_positions, on="target_bc", suffixes=("_source", "_target"))

        # Select a subset of `source_bc` for visualization (adjust number as needed)
        num_to_plot = 10  # Adjust for performance
        source_subset = df_final["source_bc"].unique()[:num_to_plot]  # Take first N unique source_bcs

        # Filter dataframe to only keep the selected `source_bc`s
        df_sample = df_final[df_final["source_bc"].isin(source_subset)]
        # Remove rows where "colors" column is NaN
        df_sample = df_sample.dropna(subset=["colors"])

        # Construct line segments: each source_bc is connected to **all** of its target_bcs
        lines = np.array([
            [row[["gt_x_source", "gt_y_source"]].values, row[["gt_x_target", "gt_y_target"]].values]
            for _, row in df_sample.iterrows()
        ])


        # Add LineCollection for edges
        line_collection = LineCollection(lines, color="k", linewidths=2, alpha=0.7)
        ax.add_collection(line_collection)

        # Scatter plot of `source_bc` positions
        

        # Scatter plot of `target_bc` positions (averaged)
        ax.scatter(df_avg_positions["gt_x"], df_avg_positions["gt_y"], c="b", s=3, label="Target BCs", alpha = 0.1)
        ax.scatter(df_sample["gt_x_source"], df_sample["gt_y_source"], c=df_sample["colors"], s=50, label="Source BCs", zorder =5)

        # Formatting
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.autoscale()

        plt.show()


    def plot_gt_edges(self, ax = None, ax_hist = None, type = "gt_uni_edges", additional_arguments = []):
        if not additional_arguments:
            length_threshold = None
        else:
            length_threshold = additional_arguments[0]

        self.load_cell_unipartite() 
        if ax == None or ax_hist == None:
            fig, ax = plt.subplots(figsize = (6,6))

        
        unipartite_df = self.unipartite_cell
        cell_indexed_reconstruction = self.reconstruction_summary.set_index("bc")
        cell_indexed_reconstruction["colors"] = cell_indexed_reconstruction["cell_type"].copy().map(self.colors)
        print("Extracting GT edges")
        only_gt_edges = unipartite_df[(unipartite_df["source"].isin(cell_indexed_reconstruction.index)) & (unipartite_df["target"].isin(cell_indexed_reconstruction.index))]

        cells_with_edge = list(set(only_gt_edges["source"]).union(set(only_gt_edges["target"])))
        positions_cells_with_edges = cell_indexed_reconstruction.loc[cells_with_edge, :]
        
        ax.set_box_aspect(1)
        print("Creating positions")
        edges_point_format = []
        source_positions = positions_cells_with_edges.loc[only_gt_edges["source"], ["gt_x", "gt_y"]].values
        target_positions = positions_cells_with_edges.loc[only_gt_edges["target"], ["gt_x", "gt_y"]].values
        print("calculating distances")
        edge_lengths = np.linalg.norm(source_positions - target_positions, axis=1)

        if length_threshold:
            valid_edges_mask = edge_lengths <= length_threshold  # Keep only edges shorter than threshold

            # Apply filtering
            source_positions = source_positions[valid_edges_mask]
            target_positions = target_positions[valid_edges_mask]
            edge_lengths = edge_lengths[valid_edges_mask]  # (Optional: keep track of valid lengths)

        # Normalize edge lengths for colormap mapping
        norm = mcolors.Normalize(vmin=min(edge_lengths), vmax=max(edge_lengths))
        cmap = cm.viridis  # Choose colormap

        # Map edge lengths to colors
        colors = cmap(norm(edge_lengths))
        threshold = 100

        # Stack the source and target positions together to create edge line segments
        if length_threshold !=0:
            normalized_lengths = norm(edge_lengths)
            edges_point_format = np.stack([source_positions, target_positions], axis=1)
            alpha_length = (1-(normalized_lengths))**4
            
            short_alpha_values = np.where(
                normalized_lengths*edge_lengths.max() < threshold,
                0.8,  # Fully opaque below threshold
                0.05  # Exponential drop-off after threshold
            )
            thicknesses = np.where(
                normalized_lengths*edge_lengths.max() < threshold,
                0.2,  # Fully opaque below threshold
                0.1  # Exponential drop-off after threshold
            )
            short_lines = LineCollection(edges_point_format, colors="k", linewidth=thicknesses, alpha=short_alpha_values)
            # # short_lines = LineCollection(edges_point_format, colors=colors, linewidth=1, alpha=long_alpha_values)

            # # Add edge collection to the top scatter plot
            print("Plotting edges")
            ax.add_collection(short_lines)
        # ax.autoscale()
        ax.set_title(f"Edge Network with Length-based Coloring:{len(edge_lengths)} edges")

        # Add colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Edge Length")

        from scipy.spatial.distance import pdist
        all_cell_positions = positions_cells_with_edges.loc[:,["gt_x", "gt_y"]].values
        all_possible_distance = pdist(all_cell_positions)
        max_bin_value = max(all_possible_distance.max(), edge_lengths.max())

        # Define consistent bin edges (same for both histograms)
        num_bins = 300
        bin_edges = np.linspace(0, max_bin_value, num_bins + 1)
        counts_all, bin_edges, patches = ax_hist.hist(all_possible_distance, bins=bin_edges, alpha = 0.5, facecolor = "lightgray")
        # Compute histogram bins and colors
        counts_edges, bin_edges, patches = ax_hist.hist(edge_lengths, bins=bin_edges, alpha=1)

        ax_ratio = ax_hist.twinx()
        ax_ratio.set_box_aspect(1)
        
        hist_ratio = np.divide(counts_edges, counts_all, where=counts_all > 0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

        # ax_ratio.plot(bin_centers, hist_ratio, color="red", linewidth=2, label="Ratio")
        ax_ratio.scatter(bin_centers, hist_ratio, color="red", s=2, label="Ratio")
        ax_ratio.set_ylim([0,1.05])
        ax_ratio.set_xlim(0,1000) 
        ax_ratio.set_box_aspect(1)
        # Apply color to histogram bins based on the colormap
        for patch, left_edge in zip(patches, bin_edges[:-1]):
            color = cmap(norm(left_edge))  # Map bin left edge to color
            patch.set_facecolor(color)

        # Labels and titles for histogram
        ax_hist.set_xlabel("Edge Length")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Histogram of Edge Lengths with Color-mapped X-axis")
        ax_hist.axvline(threshold)
        # ax_hist.set_xlim([-10, bin_edges[-1]*1.05])
        ax_hist.set_box_aspect(1)
        
        ax.scatter(positions_cells_with_edges["gt_x"], positions_cells_with_edges["gt_y"], c = positions_cells_with_edges["colors"], edgecolors="k", linewidths=0.2, s = 20, zorder = 5)
        ax.set_aspect("equal")
        fig, ax = plt.subplots(figsize=(8, 6))
        zoom_factor = self.max_gt_dist/500
        ax.scatter(positions_cells_with_edges["gt_x"], positions_cells_with_edges["gt_y"], c = positions_cells_with_edges["colors"], edgecolors="k", linewidths=0.2*zoom_factor, s = 15* zoom_factor, zorder = 5)
        short_lines = LineCollection(edges_point_format, colors="k", linewidth=thicknesses*zoom_factor/2, alpha=short_alpha_values)
        ax.set_aspect("equal")
        # # short_lines = LineCollection(edges_point_format, colors=colors, linewidth=1, alpha=long_alpha_values)

        # # Add edge collection to the top scatter plot
        print("Plotting edges")
        ax.add_collection(short_lines)
        ax.set_ylim(2100, 2600)
        ax.set_xlim(2500, 3000)

    def calculate_recon_dbscan_clusters(self, eps_percent = 5, additional_arguments = [], plot_clusters = False):
        if not additional_arguments:
            min_samples = range(5, 21)
            eps_percent = [5]
        elif len(additional_arguments)==1:
            min_samples= additional_arguments[0]
        elif len(additional_arguments)==2:
            min_samples= additional_arguments[0]
            eps_percent= additional_arguments[1]
        elif len(additional_arguments)==3:
            min_samples= additional_arguments[0]
            eps_percent= additional_arguments[1]
            plot_clusters = additional_arguments[2]

        edgelist = self.edgelist
        print(edgelist)
        edgelist_dict = edgelist.set_index(["source_bc", "target_bc"])["nUMI"].to_dict()
        df_all_recon = self.full_reconstruction_summary.copy()
        for eps in eps_percent:
            for ms in min_samples:
                df_all_recon[f"dbscan_clusters_ms={ms}_eps={eps}"] = -1
        df_all_recon.set_index("node_ID")
        reconstruction_to_analyse = 1
        recon_df = df_all_recon.loc[:, ["node_ID", "node_bc", "node_type", f"recon_x_{reconstruction_to_analyse}", f"recon_y_{reconstruction_to_analyse}", "type_prediction_score"]]
        recon_df = recon_df.rename(columns={f"recon_x_{reconstruction_to_analyse}": "x", f"recon_y_{reconstruction_to_analyse}": "y"})
        cells = recon_df[(recon_df["node_type"]!="bead")]
        max_dist = np.max(pdist(cells.loc[:, ["x", "y"]].values))

        for idx, (node_ID, cell, type, x, y, prediction_score) in cells.iterrows():
            print(f"clustering cell {idx}")
            for eps in eps_percent:
                for ms in min_samples:

                    if type =="bead":
                        break
                    # elif prediction_score==-1 or type!="B_germinal_center":
                    #     continue 

                    eps_dist = max_dist*eps/100

                    cell_edges = edgelist[edgelist["source_bc"] == cell]
                    cell_bead_positions = recon_df[recon_df["node_bc"].isin(cell_edges["target_bc"])].copy()
                    cell_edges.set_index(["source_bc", "target_bc"], inplace=True)
                    
                    cell_bead_positions["nUMI"] = cell_bead_positions["node_bc"].map(
                            lambda bc: edgelist_dict.get((cell, bc), None)  # Get nUMI using (source_bc, target_bc) tuple
                        )
                    X = cell_bead_positions[['x', 'y']].values  # Convert to a NumPy array
                    dbscan = DBSCAN(eps=eps_dist, min_samples=ms)
                    df = cell_bead_positions.copy()
                    
                    df['cluster'] = dbscan.fit_predict(X)

                    norm = mcolors.Normalize(vmin=df["nUMI"].min(), vmax=df["nUMI"].max())
                    cmap = cm.viridis

                    # Visualize the clusters
                    unique_labels = set(df['cluster'])
                    clusters = [cluster for cluster in unique_labels if cluster !=-1]
                    df_all_recon.at[idx, f"dbscan_clusters_ms={ms}_eps={eps}"] = len(clusters)

                    if not plot_clusters:
                        continue
                    # if type=="B-germinal_center":
                    #     continue

                    if len(unique_labels) != 2:
                        cluster_cmap = plt.colormaps.get_cmap("tab10")

                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.set_box_aspect(1)

                        # Optionally, plot background points from recon_df in gray
                        ax.scatter(recon_df["x"], recon_df["y"], s=5, c="gray", label="All Beads")

                        unique_labels = set(df['cluster'])

                        # Loop over clusters
                        for k in unique_labels:
                            class_member_mask = (df['cluster'] == k)
                            xy = df[class_member_mask]
                            
                            if k == -1:
                                # Noise points: use a fixed edge color (black) for noise
                                ax.scatter(
                                    xy['x'], xy['y'],
                                    c=xy['nUMI'], cmap=cmap, norm=norm,
                                    edgecolor="black", s=50, label="Noise"
                                )
                            else:
                                # Use the cluster_cmap to determine the edge color based on cluster label
                                edge_color = cluster_cmap(int(k) % cluster_cmap.N)
                                ax.scatter(
                                    xy['x'], xy['y'],
                                    c=xy['nUMI'], cmap=cmap, norm=norm,
                                    edgecolor=edge_color, s=50, label=f"Cluster {k}"
                                )

                        # Mark the "cell" position (if desired) with a distinct marker
                        # (Assuming variables x and y represent the coordinates of the cell)
                        ax.scatter(x, y, facecolors='none', edgecolors='m',  s= 100, label="Cell")

                        # Add a colorbar based on nUMI
                        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax, shrink =0.6)
                        cbar.set_label("nUMI")

                        ax.set_title(f'DBSCAN Clustering: {cell} - {type}, {prediction_score}')
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.legend()
                        plt.show()
        self.full_reconstruction_summary = df_all_recon
    
    def calculate_per_gt_cell_full_metrics(self):
        selected_columns = self.reconstruction_summary[[col for col in self.reconstruction_summary.columns if col.startswith("distortion")]]
        self.reconstruction_summary["mean_distortion"] = selected_columns.mean(axis = 1)
        selected_columns = self.reconstruction_summary[[col for col in self.reconstruction_summary.columns if col.startswith("knn")]]
        self.reconstruction_summary["mean_knn"] = selected_columns.mean(axis = 1)

        selected_columns = self.reconstruction_summary[[col for col in self.reconstruction_summary.columns if col.startswith("morphed_distortion")]]
        self.reconstruction_summary["mean_morphed_distortion"] = selected_columns.mean(axis = 1)
        selected_columns = self.reconstruction_summary[[col for col in self.reconstruction_summary.columns if col.startswith("morphed_knn")]]
        self.reconstruction_summary["mean_morphed_knn"] = selected_columns.mean(axis = 1)

        self.max_gt_dist = np.max(pdist(self.reconstruction_summary.loc[:, ["gt_x", "gt_y"]].values))

        self.reconstruction_summary["mean_edge_distance"] = None
        print(self.edgelist)
        mean_distance_map = self.edgelist.groupby("source_bc")["mean_distance"].mean()
        self.reconstruction_summary["std_edge_distance"] = None
        mean_distance_map = self.edgelist.groupby("source_bc")["std_distance"].std().fillna(0)

        # Use .map() to efficiently assign values
        self.reconstruction_summary["mean_edge_distance"] = self.reconstruction_summary["bc"].map(mean_distance_map)

    def find_modified_full_reconstruction(self):
        if self.gating_threshold != "ungated":
            matching_files = [file for file in os.listdir(self.file_location) if f"gated_{self.gating_threshold}.csv" in file and "modified_reconstruction_summary" in file]
        else:
            matching_files = [file for file in os.listdir(self.file_location) if f"_gated_ungated"  in file and "modified_reconstruction_summary" in file]
        if len(matching_files)==1:
            return pd.read_csv(f"{self.file_location}/{matching_files[0]}")
        else:
            raise Exception(f"0 or more than one file found for {self.name}: {matching_files}")

    def find_full_reconstruction(self):

        if self.gating_threshold != "ungated":
            if self.config.modification_type == "gated":
                print(os.listdir(self.file_location))
                matching_files = [file for file in os.listdir(self.file_location) if f"gated_{self.gating_threshold}" in file and "full_reconstruction_summary" in file]
            elif self.config.modification_type == "dbscan":
                matching_files = [file for file in os.listdir(self.file_location) if f"dbscan" in file and self.gating_threshold[:-3] in file and "full_reconstruction_summary" in file and ".csv" in file]
        else:
            matching_files = [file for file in os.listdir(self.file_location) if f"_dbscan_" not in file and f"_gated_" not in file and "full_reconstruction_summary" in file and "+" not in file]
        print(matching_files)
        print(self.gating_threshold)
        if len(matching_files)==1:
            return pd.read_csv(f"{self.file_location}/{matching_files[0]}")
        else:
            raise Exception(f"0 or more than one file found for {self.name}: {matching_files}")

    def find_reconstruction_summary(self):
        print(self.file_location)
        if self.gating_threshold != "ungated":
            if self.config.modification_type == "gated":
                matching_files = [file for file in os.listdir(self.file_location) if f"gated_{self.gating_threshold}.csv" in file and "reconstruction_quality_gt" in file]
            elif self.config.modification_type == "dbscan":
                matching_files = [file for file in os.listdir(self.file_location) if f"dbscan" in file and self.gating_threshold[:-3] in file and "reconstruction_quality_gt" in file and ".csv" in file]
        else:
            matching_files = [file for file in os.listdir(self.file_location) if f"_dbscan_" not in file and f"_gated_" not in file and "reconstruction_quality_gt" in file and "+" not in file]
        if len(matching_files)==1:
            return pd.read_csv(f"{self.file_location}/{matching_files[0]}")
        else:
            raise Exception(f"0 or more than one file found for {self.name}: {matching_files}")
        
    def plot_reconstruction_metric_correlation(self,ax = None,  additional_arguments = [], bins = 100, type = "mean"):
        if not additional_arguments:
            metric_1 = "knn"
            metric_2 = "distortion"
        else:
            metric_1 = additional_arguments[0]
            metric_2 = additional_arguments[1]
        if ax == None:
            fig, ax = plt.subplots(figsize = (6,6))

        x_data = self.reconstruction_summary[f"{type}_{metric_1}"]
        y_data = self.reconstruction_summary[f"{type}_{metric_2}"]
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
        except:
            slope, intercept, r_value, p_value, std_err = 1, 0, 0, 0, 0
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        x = np.linspace(x_data.min(), x_data.max(), 100)
        y = slope * x + intercept
        ax.plot(x, y, color="red", label=f"Fit (R²={r_value**2:.2f})")
        ax.scatter(x_data, y_data, s = 4)
        ax.set_xlabel(f"{type} {metric_1}")
        ax.set_ylabel(f"{type} {metric_2}")
        ax.legend()

    def plot_gt_reconstruction_metric_distribution(self, ax = None, metric = "knn", additional_arguments = []):
        
        if ax == None:
            fig, ax = plt.subplots(figsize = (6,6))

        if not additional_arguments:
            metric_str = f"{metric}"
        else:
            metric_str = f"morphed_{metric}"

        per_reconstruction_metrics = [col for col in self.reconstruction_summary.columns if col.startswith(metric_str)]
        print(per_reconstruction_metrics)
        all_means = []
        non_norm_mean = []
        total_counts = pd.Series(dtype=int)
        for reconstruction in per_reconstruction_metrics:
            per_recon_data_non_norm = self.reconstruction_summary[reconstruction]
            per_recon_data = self.reconstruction_summary[reconstruction]/self.max_gt_dist
            x = np.linspace(0, 1, 1000)
            try:
                kde = gaussian_kde(per_recon_data)
                kde_y = kde(x)
            except:
                kde_y = per_recon_data
                x = kde_y
            all_means.append(per_recon_data.median())
            if metric == "knn":
                print(per_recon_data.value_counts().sort_index())
                qa_counts = per_recon_data_non_norm.value_counts().sort_index()
                total_counts = total_counts.add(qa_counts, fill_value=0)
                total_counts = total_counts.sort_index()
                non_norm_mean.append(np.mean(per_recon_data_non_norm))
                ax.scatter(qa_counts.index, qa_counts.values, label=f"µ: {per_recon_data.mean():.4f}", s=10, c = "k", alpha =0.5)
            else:
                num_bins = 100
                
                bin_edges = np.linspace(0, 2000, num_bins + 1)
                print(per_recon_data_non_norm)
                non_norm_mean.append(np.median(per_recon_data_non_norm))
                counts, bin_edges, patches = ax.hist(per_recon_data_non_norm, bins=bin_edges, alpha=0.8)

                # Normalize to a 0–1000 range
                norm = mcolors.Normalize(vmin=0, vmax=1000)
                cmap = cm.get_cmap("magma_r")  # or "plasma", "inferno", etc.

                # Color each patch (bar) based on bin center
                for count, patch, left, right in zip(counts, patches, bin_edges[:-1], bin_edges[1:]):
                    bin_center = (left + right) / 2
                    color = cmap(norm(bin_center))
                    patch.set_facecolor(color)
            # ax.plot(x, kde_y, label=f"µ: {per_recon_data.mean():.3f}", color='m', linestyle='-', linewidth=1)
        # quit()
        print(non_norm_mean)
        # ax.set_xlim([-0.1, 1.1])
        if metric =="knn":
            
            total_sum = (total_counts.index.to_numpy() * total_counts.values).sum()
            total_n = total_counts.values.sum()
            total_mean = total_sum / total_n
            label = f"mean: {total_mean:.4f}"
            
            values = total_counts.index.to_numpy()
            counts = total_counts.values
            variance = ((counts * (values - total_mean) ** 2).sum()) / total_n
            total_std = np.sqrt(variance)
            print(f"mean: {total_mean}±{total_std}")
            print(f"per run mean {np.mean(non_norm_mean)}±{np.std(non_norm_mean)}")
            ax.plot(total_counts.index, total_counts.values/len(per_reconstruction_metrics))
        elif metric =="distortion":
            total_mean = np.mean(all_means)
            total_std = np.std(all_means)
            metric = "relative distortion"
            label = f"mean median: {total_mean:.3f}±{total_std:.3f}({np.mean(non_norm_mean):.1f}±{np.std(non_norm_mean):.1f}µm)"
        ax.axvline(np.mean(non_norm_mean), c = "r", linestyle = "--", label = label)
        
        # ax.vlines(data, ymin=0, ymax=10, colors="black", lw=0.5, alpha = 0.5)
        ax.set_xlabel(f"{metric}")
        ax.set_ylabel("frequency")
        ax.legend()

    def plot_knn_over_k(self, ax=None, additional_arguments = []):
        print(self.full_reconstruction_summary)
        print(self.reconstruction_summary)
        from scipy.spatial import KDTree
        if ax == None:
            fig, ax = plt.subplots(figsize = (6,6))

        if not additional_arguments:
            col_selection = f"align_recon"
            single = False
        else:
            if "morph" in additional_arguments:
                col_selection = f"align_morph_recon"
            else:
                col_selection = f"align_recon"
            if "single" in additional_arguments:
                single=True
            else:
                single = False

        gt_positions = self.reconstruction_summary[["bc", "gt_x", "gt_y"]].copy().set_index("bc")
        reconstructed_positions = self.full_reconstruction_summary.copy().set_index("node_bc")
        reconstructed_positions = reconstructed_positions[[col for col in reconstructed_positions.columns if col.startswith(col_selection)]]

        n_reconstructions = int(len(reconstructed_positions.columns)/2)
        matching_indexes = gt_positions.index.intersection(reconstructed_positions.index)
        gt_for_knn = gt_positions.loc[matching_indexes]
        
        all_knns_over_k = []
        k_list = range(2, 50, 1)
        for k in k_list:
            print(k)            
            original_tree = KDTree(gt_for_knn)
            original_neighbors = original_tree.query(gt_for_knn, k + 1)[1][:, 1:]
            mean_knn_per_recon = []
            for i in range(n_reconstructions):
                recon_with_gt = reconstructed_positions[[f"{col_selection}_x_{i+1}", f"{col_selection}_y_{i+1}"]]
                recon_with_gt = recon_with_gt.loc[matching_indexes]
                reconstructed_tree = KDTree(recon_with_gt)
                reconstructed_neighbors = reconstructed_tree.query(recon_with_gt, k + 1)[1][:, 1:]
                knn_per_point = []
                for original, reconstructed in zip(original_neighbors, reconstructed_neighbors):               
                    n = len(original)
                    knn_per_point.append(len(set(original).intersection(set(reconstructed[:n]))) / n)
                    if single:
                        break

                print(n)
                mean_knn_per_recon.append(np.mean(knn_per_point))

                ax.scatter(k, np.mean(knn_per_point), color='k', alpha = 0.3, s =5)
                if single:
                    break

            all_knns_over_k.append(np.mean(mean_knn_per_recon))

        ax.plot(k_list, all_knns_over_k)
        # ax.axvline(total_mean, c = "r", linestyle = "--", label = label)
        ax.set_ylim([0, 1.1])
        ax.axhline(1, c = "r", linestyle = "--")
        # ax.vlines(data, ymin=0, ymax=10, colors="black", lw=0.5, alpha = 0.5)
        ax.set_xlabel(f"K")
        ax.set_ylabel("mean knn value")
        ax.set_box_aspect(1)

    def plot_no_gt_metric_correlation(self, ax=None, metric_1 = "mean", metric_2 = "std"):
        
        if self.n_reconstructions <3 and (metric_1=="std" or metric_2 =="std" or metric_1=="max_diff" or metric_2 =="max_diff"):
            return
        if ax == None:
            fig, ax = plt.subplots(1, 1)
        edges = self.edgelist
        try:
            slope, intercept, r_value, p_value, std_err = linregress(edges[f"{metric_1}_distance"], edges[f"{metric_2}_distance"])
        except:
            slope, intercept, r_value, p_value, std_err = 1, 0, 0, 0, 0
        # Generate the line
        x = np.linspace(edges[f"{metric_1}_distance"].min(), edges[f"{metric_1}_distance"].max(), 100)
        y = slope * x + intercept
        ax.plot(x, y, color="red", label=f"Fit (R²={r_value**2:.2f})")

        ax.scatter(edges[f"{metric_1}_distance"], edges[f"{metric_2}_distance"], s = 3)
        ax.set_xlabel(f"distance {metric_1}")
        ax.set_ylabel(f"distance {metric_2}")
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        ax.legend()

        ax.set_title(f"reconstructed_distance_variance\n{len(edges)} edges")

    def plot_distance_distribution(self, ax=None, data_type = "mean"):
        # fi g, ax = plt.subplots(1, 1)
        if ax == None:
            fig, ax = plt.subplots(1, 1)
        
        data_base = self.edgelist.loc[self.edgelist[f"{data_type}_distance"]!=0, f"{data_type}_distance"]
        log_data = np.log(data_base)
        # data_base=log_data
        print(data_base)
        # data_base = data_base.loc[data_base[f"{data_type}_distance"]!=0]

        # data_base=log_data
        if data_base.isnull().any():
            return
        ax.hist(data_base, bins = len(data_base)%2000)
        ax.set_xlabel(f"Reconstructed distance {data_type}")
        ax.set_ylabel("Frequency")
        # ax.set_title(f"reconstructed_distance_distribution\n{len(data_base)} edges")
        # gmm = GaussianMixture(n_components=5, random_state=2)
        # data = data_base.values.reshape(-1, 1)  # GMM requires data in 2D
        # gmm.fit(data)

        # # Extract fitted parameters
        # means = gmm.means_.flatten()  # Means of the Gaussians
        # variances = gmm.covariances_.flatten()  # Variances of the Gaussians
        # weights = gmm.weights_  # Mixing coefficients
        # x = np.linspace(data.min(), data.max(), 1000)
        # for mean, variance, weight, i in zip(means, variances, weights, range(len(means))):
        #     y = weight * (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-0.5 * ((x - mean)**2 / variance))
        #     ax.plot(x, y, label=f"G{i+1} (μ={mean:.2f}, σ={np.sqrt(variance):.2f})")

        # Plot the combined PDF
        # combined_pdf = np.sum([
        #     weight * (1 / (np.sqrt(2 * np.pi * variance))) * np.exp(-0.5 * ((x - mean)**2 / variance))
        #     for mean, variance, weight in zip(means, variances, weights)
        # ], axis=0)
        # ax.plot(x, combined_pdf, label="sum G", color='red', linestyle='--', linewidth = 0.5)

        # data_select = random.sample(range(0, len(data.flatten())), len(data.flatten()//10))

        # data = data.flatten()[data_select]

        # kde = gaussian_kde(data)
        # kde_y = kde(x)
        # local_maxima = argrelextrema(kde_y, np.greater)[0]
        # local_minima = argrelextrema(kde_y, np.less)[0]
        # ax.plot(x, kde_y, label="KDE", color='m', linestyle='-', linewidth=3)
        # plt.scatter(x[local_maxima], kde_y[local_maxima], color='red', label='Maxima', zorder=5)
        # plt.scatter(x[local_minima], kde_y[local_minima], color='green', label='Minima', zorder=5)

        # lower_mean_index = np.argmin(means)  # Index of the Gaussian with the lower mean
        # mean = means[lower_mean_index]
        # std_dev = np.sqrt(variances[lower_mean_index])  # Standard deviation

        # # Calculate the 99th percentile
        # from scipy.stats import norm
        # percentile_99 = norm.ppf(0.999, loc=mean, scale=std_dev)
        # ax.axvline(percentile_99, label =f"Spatial_max:{percentile_99:.2f}", linestyle ="--", c = "blue")
        # ax.legend()
            
        if False:
            # ax.set_yscale("log")

            # ax.set_xscale("log")
            from scipy.stats import nbinom
            from scipy.optimize import minimize
            def negative_binomial_pmf(x, n, p):
                return nbinom.pmf(x, n, p)

            # Negative log-likelihood for fitting
            def neg_log_likelihood(params, data):
                n, p = params
                return -np.sum(np.log(negative_binomial_pmf(data, n, p) + 1e-9))
            
            data = data_base.values

            # Initial guesses for parameters n and p
            initial_guess = [5, 0.5]
            bounds = [(1e-3, None), (1e-3, 1 - 1e-3)]

            # Optimize parameters
            result = minimize(neg_log_likelihood, initial_guess, args=(data,), bounds=bounds)
            n_fitted, p_fitted = result.x

            # Generate x values and the fitted negative binomial PDF
            x = np.arange(data.min(), data.max() + 1)
            fitted_pdf = nbinom.pmf(x, n_fitted, p_fitted)
            ax.plot(x, fitted_pdf, color="red", label=f"Negative Binomial Fit\nn={n_fitted:.2f}, p={p_fitted:.2f}")
            ax.legend()

    def plot_cell_type_distance_distribution(self, ax = None):
        detailed_edgelist = self.edgelist
        import seaborn as sns
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        fig = plt.figure()
        
        categories_to_include = (
            detailed_edgelist.loc[detailed_edgelist["source_type"] != "bead"]
            .groupby("source_type")["distance_1"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        category_counts = detailed_edgelist["source_type"].value_counts()

        unique_cell_counts = detailed_edgelist.groupby("source_type")["source_bc"].nunique()


        normalized_counts = unique_cell_counts / unique_cell_counts.max()
        norm = Normalize(vmin=unique_cell_counts.min(), vmax=unique_cell_counts.max())
        colormap = plt.cm.viridis

        # Map categories to colors based on normalized counts
        category_colors = {
            category: colormap(normalized_counts[category])
            for category in categories_to_include
        }
        # Create the boxplot, specifying the order of categories to include
        
        sns.boxplot(
            x="source_type",
            y="distance_1",
            data=detailed_edgelist,
            order=categories_to_include,
            showfliers=False,
            hue="source_type",  # Assign the x-variable to hue
            palette=category_colors,  # Apply the custom color mapping
            dodge=False,  # Prevent splitting into multiple boxes (since hue == x)
            legend=False  # Remove unnecessary legend
        )
        plt.xlabel("cell type")
        plt.suptitle(f"{len(detailed_edgelist.index)} edges")
        plt.xticks(rotation=45, fontsize = 8)

        xticks = plt.gca().get_xticks()  # Get current tick positions

        xticklabels = plt.gca().get_xticklabels()  # Get current tick labels
        for tick, label in zip(xticks, xticklabels):
            label.set_ha('right')  # Align horizontally to the right

        sm = ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])  # Required for ScalarMappable to work with colorbar
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("N cells per category")
        plt.tight_layout()
    
    def analyse_dbscan_clusters(self, ax = None, ms = [5]):
        if not additional_arguments:
            min_samples = range(5, 21)
            eps_percent = [5]
        elif len(additional_arguments)==1:
            min_samples= additional_arguments[0]
        elif len(additional_arguments)==2:
            min_samples= list(additional_arguments[0])
            eps_percent= list(additional_arguments[1])

        n_gt_cells_single_cluster = []
        n_no_gt_cells_single_cluster = []
        n_total_cells_single_cluster = []

        print(eps_percent)
        print(min_samples)
        for i, eps in enumerate(eps_percent):
            n_min_samples = len(min_samples)
            max_columns = 7
            if n_min_samples<max_columns:
                max_columns =n_min_samples

            n_rows = (n_min_samples + max_columns - 1) // max_columns  # Calculate the required rows
            
            fig, axes = plt.subplots(n_rows, max_columns, figsize=(max_columns * 3, n_rows * 3))

            axes = np.array(axes)  # Ensure it's an array for indexing
            axes = axes.flatten()  # Flatten to handle single indexing
            plt_index = 0
            for j, ms in enumerate(min_samples):
                ax = axes[plt_index]
                plt_index +=1
                all_nodes = self.find_modified_full_reconstruction()
                all_cells = all_nodes.loc[all_nodes["node_type"]!="bead", :]
                clustered_cells = all_nodes.loc[all_nodes[f"dbscan_clusters_ms={ms}_eps={eps}"]!=-1, :]

                no_gt_cells = all_cells.loc[all_cells["type_prediction_score"]!=-1, :]
                counts_no_gt = no_gt_cells[f"dbscan_clusters_ms={ms}_eps={eps}"].value_counts().sort_index()
                counts_df_no_gt = counts_no_gt.reset_index()
                counts_df_no_gt.columns = [f"dbscan_clusters_ms={ms}_eps={eps}", "Frequency_no_gt"]

                # Get gt cells (type_prediction_score == -1)
                gt_cells = all_cells.loc[all_cells["type_prediction_score"] == -1, :]
                counts_gt = gt_cells[f"dbscan_clusters_ms={ms}_eps={eps}"].value_counts().sort_index()
                counts_df_gt = counts_gt.reset_index()
                counts_df_gt.columns = [f"dbscan_clusters_ms={ms}_eps={eps}", "Frequency_gt"]

                # Merge the two DataFrames on the cluster column (outer merge to keep all groups) and fill missing values with 0
                merged_counts = pd.merge(counts_df_no_gt, counts_df_gt, on=f"dbscan_clusters_ms={ms}_eps={eps}", how="outer").fillna(0)
                merged_counts = merged_counts.sort_values(f"dbscan_clusters_ms={ms}_eps={eps}")

                # Create a stacked bar plot
                
                merged_counts_modified = merged_counts.set_index(f"dbscan_clusters_ms={ms}_eps={eps}")
                # print(merged_counts_modified)
                if 1 in merged_counts_modified.index:
                    single_cluster_count_gt = merged_counts_modified.loc[1, "Frequency_gt"]
                    single_cluster_count_no_gt = merged_counts_modified.loc[1, "Frequency_no_gt"]
                    single_cluster_count_total = merged_counts_modified.loc[1, :].sum()
                else:
                    single_cluster_count_gt, single_cluster_count_no_gt, single_cluster_count_total =0,0,0

                n_total_cells_single_cluster.append([ms, eps, single_cluster_count_total])
                n_gt_cells_single_cluster.append([ms, eps, single_cluster_count_gt])
                n_no_gt_cells_single_cluster.append([ms, eps, single_cluster_count_no_gt])
                # Plot no_gt frequencies first
                ax.bar(
                    merged_counts[f"dbscan_clusters_ms={ms}_eps={eps}"],
                    merged_counts["Frequency_no_gt"],
                    label=f"{100*single_cluster_count_no_gt/len(no_gt_cells):.2f}% No GT 1 cluster"
                )

                # Plot gt frequencies, stacking them on top of the no_gt frequencies
                ax.bar(
                    merged_counts[f"dbscan_clusters_ms={ms}_eps={eps}"],
                    merged_counts["Frequency_gt"],
                    bottom=merged_counts["Frequency_no_gt"],
                    label=f"{100*single_cluster_count_gt/len(gt_cells):.2f}% GT 1 cluster"
                )

                ax.set_xlabel(f"dbscan_clusters_ms={ms}_eps={eps}")
                ax.set_ylabel("Frequency") 
                ax.set_title(f"ms={ms} eps = {eps}, 1 cluster cells: {100*single_cluster_count_total/len(all_cells):.2f}")
                ax.legend()
                ax.set_xticks(np.arange(np.min(merged_counts[f"dbscan_clusters_ms={ms}_eps={eps}"]), np.max(merged_counts[f"dbscan_clusters_ms={ms}_eps={eps}"])+1, 1))

                plt.tight_layout()
            # plt.show()
        print(n_total_cells_single_cluster)
        print(n_gt_cells_single_cluster)
        print(n_no_gt_cells_single_cluster)
        df1 = pd.DataFrame(n_total_cells_single_cluster, columns=["min_sample", "eps%", "total_single_cluster_cells"])
        df2 = pd.DataFrame(n_total_cells_single_cluster, columns=["min_sample", "eps%", "gt_single_cluster_cells"])
        df3 = pd.DataFrame(n_no_gt_cells_single_cluster, columns=["min_sample", "eps%", "new_single_cluster_cells"])

        # Merge on the common keys: x and y
        dbscan_results = df1.merge(df2, on=["min_sample", "eps%"]).merge(df3, on=["min_sample", "eps%"])

        print(dbscan_results)
        print("Most single cluster cells")
        row_index = dbscan_results["total_single_cluster_cells"].idxmax()
        row = dbscan_results.loc[row_index]
        print(row)
        print("Most single cluster reference cells")
        row_index = dbscan_results["gt_single_cluster_cells"].idxmax()
        row = dbscan_results.loc[row_index]
        print(row)
        print("Most single cluster new cells")
        row_index = dbscan_results["new_single_cluster_cells"].idxmax()
        row = dbscan_results.loc[row_index]
        print(row)
        return fig
    
    def compare_normal_and_morphed(self, metric = "distortion", ax_normal = None, ax_morph = None, type =None):
        if ax_normal == None or ax_morph == None:
            fig, (ax_normal, ax_morph ) = plt.subplots(1, 2, figsize = (12, 6))

        print(self.reconstruction_summary)
        print(self.gating_threshold)
        morph_columns = self.reconstruction_summary[[col for col in self.reconstruction_summary.columns if col.startswith(f"morphed_{metric}")]]
        morph_statistics = []
        for i in range(len(morph_columns.columns)):
            current_data = morph_columns[f"morphed_{metric}_{i+1}"]
            if metric == "distortion":
                current_data = 100*current_data/self.max_gt_dist
                plot_metric = "relative_distortion (%)"
            if metric == "knn":
                x = np.linspace(0, 1, 1000)
                bins = 10
            else:
                x = np.linspace(current_data.min(), current_data.max(), 1000)
            ax_morph.hist(current_data.values, bins = len(current_data)%200, density=True, alpha = 0.3)
            kde = gaussian_kde(current_data)
            kde_y = kde(x)
            
            ax_morph.set_title(f"{self.gating_threshold}\nmorphed {metric}s")
            morph_statistics.append(current_data.median())
        ax_morph.axvline(np.mean(morph_statistics), c = "r", linestyle = "--", label = f"mean of means: {np.mean(morph_statistics):.4f}")
        ax_morph.plot(x, kde_y, label="KDE", color='m', linestyle='-', linewidth=1)

        ax_morph.set_xlim([0, 100])
        ax_morph.set_xlabel(f"{plot_metric}")
        ax_morph.set_ylabel("frequency")
        ax_morph.legend()

        normal_columns = self.reconstruction_summary[[col for col in self.reconstruction_summary.columns if col.startswith(f"{metric}")]]
        normal_statistics = []
        for i in range(len(normal_columns.columns)):
            current_data = normal_columns[f"{metric}_{i+1}"]
            if metric == "distortion":
                current_data = 100*current_data/self.max_gt_dist
                plot_metric = "relative_distortion (%)"
            if metric == "knn":
                x = np.linspace(0, 1, 1000)
                bins = 10
            else:
                x = np.linspace(current_data.min(), current_data.max(), 1000)
            ax_normal.hist(current_data.values, bins = len(current_data)%200, density=True, alpha = 0.3)

            kde = gaussian_kde(current_data)
            kde_y = kde(x)
            normal_statistics.append(current_data.median())
            
            ax_normal.set_title(f"{self.gating_threshold}\nunmorphed {len(current_data)}s")
        ax_normal.axvline(np.mean(normal_statistics), c = "r", linestyle = "--", label = f"mean of means: {np.mean(normal_statistics):.4f}")
        ax_normal.plot(x, kde_y, label="KDE", color='m', linestyle='-', linewidth=1)
        ax_normal.set_xlim([0, 100])
        # ax_normal.set_xlabel(f"{plot_metric}")

        ax_normal.set_ylabel("frequency")
        ax_normal.legend()
        print(morph_statistics)
        print(normal_statistics)
        if type == "scatter":
            fig, ax_scatter = plt.subplots(1, 1, figsize = (6, 6))
            ax_scatter.scatter(morph_statistics, normal_statistics)
            line = np.linspace(0, np.max(normal_statistics)*1.1, 10)
            ax_scatter.plot(line, line, c = "r")
    
    def plot_only_beads(self, ax = None, additional_arguments = [], density_coloring = False):
        if additional_arguments:
            density_coloring = additional_arguments[0]
        print(self.full_reconstruction_summary)
        if ax == None:
            fig, ax = plt.subplots(figsize = (6,6))
            ax.set_aspect("equal")

        beads = self.full_reconstruction_summary.loc[self.full_reconstruction_summary["node_type"]!="bead",:]
        x = beads["morph_recon_x_1"].values
        y = beads["morph_recon_y_1"].values

        # Compute KDE-based density for each point
        if density_coloring:
            xy = np.vstack([x, y])
            density = gaussian_kde(xy, bw_method=0.1)(xy)

            # Create a scatter plot, using density as the color
            scatter = ax.scatter(x, y, s=3, c=density, cmap="viridis", edgecolor='none')
            plt.colorbar(scatter, ax=ax, label="Point Density")
        else:
            scatter = ax.scatter(x, y, s=3, c="k", edgecolor='none', alpha = 0.5)

    def plot_cpd_scatter(self, ax = None, additional_arguments = []):
        from scipy.spatial.distance import pdist
        if not additional_arguments:
            type = "recon"
        else:
            if "morph" in additional_arguments:
                type = f"morph_recon"
            else:
                type = "recon"
            if "subsample" in additional_arguments:
                subsample = True
            else:
                subsample = False
        if not ax:
            fig, ax = plt.subplots(figsize = (6,6))

        gt_positions = self.reconstruction_summary.copy().set_index("bc")
        gt_positions = gt_positions[["gt_x", "gt_y"]]
        recon_positions = self.full_reconstruction_summary.copy().set_index("node_bc")
        recon_positions = recon_positions.loc[gt_positions.index, :]
        recon_columns = recon_positions[[col for col in recon_positions.columns if col.startswith(type)]]
        if subsample:
            sampled_barcodes = np.random.choice(gt_positions.index, size=1000, replace=False)
            gt_positions = gt_positions.loc[sampled_barcodes]
        gt_distances = pdist(gt_positions)
        x = gt_distances
        gt_distances_dict = {i: d for i, d in enumerate(gt_distances)}
        recon_distances_dict = {i: [] for i, d in enumerate(gt_distances)}
        all_cpd = []
        for i in range(len(recon_columns.columns)//2):
            recon_positions = recon_columns[[f"{type}_x_{i+1}", f"{type}_y_{i+1}"]]
            if subsample:
                recon_positions = recon_positions.loc[sampled_barcodes]
            recon_distances = pdist(recon_positions)
            correlation, _ = pearsonr(gt_distances, recon_distances)
            cpd_score = correlation**2
            all_cpd.append(cpd_score)
            print(cpd_score)
            y = recon_distances
            slope, intercept = np.polyfit(x, y, 1)  # First-degree polynomial fit (linear)
            y_pred = np.polyval([slope, intercept], x)  # Compute predicted values

            # Plot best-fit line
            # ax.plot(x, y_pred, color="red", linewidth=0.5, label=f"(R^2={cpd_score:.4f})", linestyle = ":")
            for idx, (gt_dist, recon_dist) in enumerate(zip(gt_distances, recon_distances)):
                recon_distances_dict[idx].append(recon_dist)  # Append each reconstructed distance

            # print(recon_distances_dict) 
            print(i+1)
            # break
        average_reconstructed_distances = {
            idx: np.mean(recon_list) for idx, recon_list in recon_distances_dict.items()
        }
        print(f"avg: {np.mean(all_cpd)}±{np.std(all_cpd)}")
        sorted_indices = sorted(gt_distances_dict.keys())
        sorted_gt_distances = np.array([gt_distances_dict[idx] for idx in sorted_indices])
        sorted_avg_recon_distances = np.array([average_reconstructed_distances[idx] for idx in sorted_indices])
        correlation_avg, _ = pearsonr(sorted_gt_distances, sorted_avg_recon_distances)
        cpd_score = correlation**2
        print(cpd_score)
        # return
        slope, intercept = np.polyfit(sorted_gt_distances, sorted_avg_recon_distances, 1)  # First-degree polynomial fit (linear)
        y_pred = np.polyval([slope, intercept], sorted_gt_distances)  # Compute predicted values

        # Plot best-fit line
        ax.plot(sorted_gt_distances, y_pred, color="r", linewidth=0.5, label=f"(R^2={correlation_avg:.4f})", linestyle = "--")
        bins = 100  # Adjust bin size for better density estimation

        # Compute density using a 2D histogram
        print(sorted_gt_distances, sorted_avg_recon_distances)
        density, xedges, yedges = np.histogram2d(sorted_gt_distances, sorted_avg_recon_distances, bins=bins)

        # Convert (x, y) values into corresponding density values
        x_bin_idx = np.digitize(sorted_gt_distances, xedges) - 1
        y_bin_idx = np.digitize(sorted_avg_recon_distances, yedges) - 1
        x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
        y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)
        densities = density[x_bin_idx, y_bin_idx]

        # Normalize density values for color mapping
        densities = densities / densities.max()

        # Scatter plot with density-based coloring
        if self.config.vizualisation_args.save_to_image_format =="pdf":
            np.random.seed(42)
            num_points = 100000
            num_points = min(num_points, len(x))  

            # Generate random indices
            random_indices = np.random.choice(len(x), num_points, replace=False)

            x = x[random_indices]
            y = y[random_indices]
            c = densities[random_indices]
            ax.legend(loc = "upper left")
        else: 
            c = densities
        sc = ax.scatter(sorted_gt_distances, sorted_avg_recon_distances, c=c, cmap="magma_r", s=1, edgecolors="none", alpha = 1)
        ax.set_title(f"{self.gating_threshold}, {len(gt_distances)} distances")
        plt.colorbar(sc, label="Point Density")
        # ax.scatter(gt_distances, recon_distances, s = 3)

    def per_cell_type_metrics(self, fig = None, additional_arguments = []):
        if fig:
            plt.close(fig)
            del fig
        if not additional_arguments:
            metric = f"mean_knn"
        else:
            if len(additional_arguments) ==1:
                metric = f"mean_{additional_arguments[0]}"
            else:
                metric = f"mean_morphed_{additional_arguments[0]}"

        full_recon = self.full_reconstruction_summary.copy()
        quality_metrics_and_gt = self.reconstruction_summary.copy()
        print(full_recon)
        print(self.edgelist)
        print(quality_metrics_and_gt)
        median_order = (
            quality_metrics_and_gt.groupby("cell_type")[metric]
            .median()
            .sort_values(ascending=False)
            .index
        )
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=quality_metrics_and_gt,
            x="cell_type",
            y=metric,
            order=median_order,
            inner="box",  # shows box inside the violin
            cut=0
        )
        plt.axhline(quality_metrics_and_gt[metric].median(), linestyle = "--", c = "r")
        # Style
        plt.xticks(rotation=45)
        plt.xlabel("Cell Type")
        plt.ylabel("Mean KNN")
        plt.title("KNN Distribution by Cell Type")
        plt.ylim([0, 1.1])
        plt.tight_layout()

        
    def plot_cell_types(self, fig = None, additional_arguments = [], subgraph_collection:subgraphCollection=None): 
        base_subgraph = subgraph_collection.unmodified_subgraph
        if fig:
            plt.close(fig)
            del fig
        # print(self.full_reconstruction_summary)
        base_full_recon = base_subgraph.find_modified_full_reconstruction()
        base_quality_metrics_and_gt = base_subgraph.reconstruction_summary.copy().reset_index().set_index("cell_type")

        full_recon = self.full_reconstruction_summary.copy()
        quality_metrics_and_gt = self.reconstruction_summary.copy().reset_index().set_index("cell_type")

        full_recon.set_index("node_type", inplace=True)
        base_full_recon.set_index("node_type", inplace=True)

        n_cell_types = len(full_recon.index.unique())
        cell_types = full_recon.index.unique()
        from scipy.stats import gaussian_kde
        for i, cell_type in enumerate(cell_types):
            
            if cell_type=="unknown_cell" or cell_type =="bead": #cell_type !="B_germinal_center"
                continue
            fig, ((ax_new_cells, ax_old_cells, ax_gt),(al1, al2, al3)) = plt.subplots(2, 3, figsize = (16, 12))
            all_typed_cells = full_recon.loc[cell_type, :].set_index("node_ID")
            gt_typed_cells = quality_metrics_and_gt.loc[cell_type, :]

            base_all_typed_cells = base_full_recon.loc[cell_type, :].set_index("node_ID")
            base_gt_typed_cells = base_quality_metrics_and_gt.loc[cell_type, :]
            print(all_typed_cells)
            ms = 12
            dbscan_values = base_all_typed_cells[[col for col in base_all_typed_cells.columns if col.startswith(f"dbscan_clusters_ms={ms}_eps={eps}")]]
            norm = plt.Normalize(0, vmax=dbscan_values.max())
            cmap = plt.cm.plasma
            
            gt_cells = all_typed_cells[all_typed_cells["type_prediction_score"]==-1]
            gt_cells_dbscan_values = base_all_typed_cells.loc[gt_cells.index, f"dbscan_clusters_ms={ms}_eps={eps}"]
            new_cells = all_typed_cells[all_typed_cells["type_prediction_score"]!=-1]
            new_cells_dbscan_values = base_all_typed_cells.loc[new_cells.index, f"dbscan_clusters_ms={ms}_eps={eps}"]

            old_cell_points = gt_cells[["align_morph_recon_x_1", "align_morph_recon_y_1"]].values
            new_cell_points = new_cells[["align_morph_recon_x_1", "align_morph_recon_y_1"]].values
            from scipy.spatial import cKDTree
            # Build KDTree for old cells
            tree = cKDTree(old_cell_points)

            # Define a search radius (e.g., 20 units)
            threshold_neighbors = 10
            distances, _ = tree.query(old_cell_points, k=threshold_neighbors + 1)  # +1 because the closest point is itself
            # Compute the average distance to the threshold closest points (excluding self)
            avg_radius = np.mean(distances[:, 1:threshold_neighbors + 1], axis=1)  # Ignore first column (self-distance)
            avg_radius = np.mean(avg_radius)
            radius = np.max([avg_radius, 50])

            num_neighbors = np.array([len(tree.query_ball_point(p, radius)) for p in new_cell_points])

            # Define a threshold for "high density" (e.g., at least 5 old_cells within radius)
            threshold = 1
            high_density_mask = num_neighbors >= threshold  # Boolean mask
            num_fulfilling = np.sum(high_density_mask)


            # Assign colors: "red" for close points, colormap for others
            colors = np.where(high_density_mask, "m", "lightgray")  
            sizes = np.where(high_density_mask, 15, 5)  # Larger (15) if close, smaller (5) if far


            ax_new_cells.scatter(new_cells["align_morph_recon_x_1"], new_cells["align_morph_recon_y_1"], s = sizes, c=colors)
            ax_new_cells.set_aspect("equal")
            ax_new_cells.set_title(f"cells close to old: {num_fulfilling}/{len(num_neighbors)}")
            ax_old_cells.scatter(gt_cells["align_morph_recon_x_1"], gt_cells["align_morph_recon_y_1"], s = 5, c="g")
            # ax_old_cells.set_box_aspect(1)
            ax_old_cells.set_aspect("equal")
            ax_gt.scatter(gt_typed_cells["gt_x"], gt_typed_cells["gt_y"], s = 3, c="g") #cmap(norm(gt_cells_dbscan_values))
            # ax_gt.set_box_aspect(1)
            ax_gt.set_aspect("equal")
            ax_gt.set_title(f"")
            ax_old_cells.set_title(f"N cells: {len(gt_typed_cells)} avg {threshold_neighbors} cell redius: {radius:.0f}")
            for (x, y), is_high_density in zip(new_cell_points, high_density_mask):
                if is_high_density:  # Draw only for cells that meet the density requirement
                    circle = plt.Circle((x, y), radius, color='gray', fill=False, linestyle='dashed', alpha=0.5)
                    ax_new_cells.add_patch(circle)
            al1.scatter(gt_cells["align_morph_recon_x_1"], gt_cells["align_morph_recon_y_1"], s = 4,c = "g", alpha = 0.5)#, alpha = norm(gt_cells_dbscan_values))
            al1.scatter(new_cells["align_morph_recon_x_1"], new_cells["align_morph_recon_y_1"], s = sizes,c = colors)#,  alpha = norm(new_cells_dbscan_values))
            # al2.set_box_aspect(1)
            al1.set_aspect("equal")


            bc_indexed_typed_cells = gt_typed_cells.copy().set_index("bc")
            bc_indexed_gt_cells = gt_cells.copy().set_index("node_bc")

            bc_indexed_gt_cells[["gt_x", "gt_y"]] = bc_indexed_typed_cells[["gt_x", "gt_y"]]

            gt_x = bc_indexed_gt_cells["align_morph_recon_x_1"].values
            gt_y = bc_indexed_gt_cells["align_morph_recon_y_1"].values

            typed_x = bc_indexed_gt_cells["gt_x"].values
            typed_y = bc_indexed_gt_cells["gt_y"].values

            # Ensure they are of the same length and correspond correctly
            assert len(gt_x) == len(typed_x), "Mismatch in corresponding point lengths!"

            # Create line segments (each pair is a line between corresponding points)
            line_segments = np.array([[ [gt_x[i], gt_y[i]], [typed_x[i], typed_y[i]] ] for i in range(len(gt_x))])

            # Create a LineCollection (efficient for plotting many lines)
            lines = LineCollection(line_segments, colors="b", linewidths=2, alpha=0.5)
            al3.add_collection(lines)
            print(gt_typed_cells)
            print(gt_cells)


            al3.scatter(bc_indexed_gt_cells["align_morph_recon_x_1"], bc_indexed_gt_cells["align_morph_recon_y_1"], s = 0, c ="g")
            al3.scatter(bc_indexed_gt_cells["gt_x"], bc_indexed_gt_cells["gt_y"], s = 0, c = "k") #cmap(norm(gt_cells_dbscan_values))
            al3.set_aspect("equal")
            plt.suptitle(f"{cell_type}")
            

            line_lengths = np.sqrt((gt_x - typed_x)**2 + (gt_y - typed_y)**2)
            
            num_bins = 20
            bin_edges = np.linspace(0, self.max_gt_dist, num_bins + 1)  # +1 to ensure 10 bins

            al2.set_xlim(0, self.max_gt_dist)
            al2.hist(line_lengths, bins=bin_edges)
            # plt.show()
    def plot_node_types(self, fig = None, additional_arguments = []):
        print(self.full_reconstruction_summary)
        if fig:
            plt.close(fig)
            del fig

        if not additional_arguments:
            metric_str = f"align_recon"
        else:
            metric_str = f"align_morph_recon"
        cell_indexed_reconstruction = self.full_reconstruction_summary
        print(cell_indexed_reconstruction)
        gt_cells = cell_indexed_reconstruction[(cell_indexed_reconstruction["type_prediction_score"]==-1) & (~cell_indexed_reconstruction["node_type"].isin(["bead", "unknown_cell"]))]
        new_cells = cell_indexed_reconstruction[(cell_indexed_reconstruction["type_prediction_score"]!=-1) | (cell_indexed_reconstruction["node_type"]=="unknown_cell")]
        beads = cell_indexed_reconstruction[(cell_indexed_reconstruction["node_type"]=="bead")]
        self.colors["unknown_cell"] = (0, 0, 0, 0.2)

        self.colors["bead"] = "k"
        gt_cells["colors"] = gt_cells["node_type"].copy().map(self.colors)
        
        print(self.colors)
        new_cells["colors"] = new_cells["node_type"].copy().map(self.colors)
        fig, (new_ax, old_ax, ax_beads) = plt.subplots(1, 3, figsize = (20, 6))
        print(new_cells)

        new_ax.scatter(new_cells[f"{metric_str}_x_1"], new_cells[f"{metric_str}_y_1"], c = new_cells["colors"], s = 5)
        new_ax.set_aspect("equal")
        #set_box_aspect(1)
        #set_aspect("equal")

        print(gt_cells)
        old_ax.scatter(gt_cells[f"{metric_str}_x_1"], gt_cells[f"{metric_str}_y_1"], c = gt_cells["colors"], s = 5)
        old_ax.set_aspect("equal")
        
        ax_beads.scatter(beads[f"{metric_str}_x_1"], beads[f"{metric_str}_y_1"], c = "k", s = 0.5, alpha = 0.4)
        ax_beads.set_aspect("equal")

    def plot_recon_dist_vs_umis(self, ax = None):
        if ax == None:
            fig, ax = plt.subplots(1, 1)
        print(self.edgelist)
        bins = 100  # Adjust bin size for better density estimation

        # Compute density using a 2D histogram
        print(self.edgelist)
        x = 1/self.edgelist["mean_distance"]
        y = self.edgelist["nUMI"]
        density, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # Convert (x, y) values into corresponding density values
        x_bin_idx = np.digitize(x, xedges) - 1
        y_bin_idx = np.digitize(y, yedges) - 1
        x_bin_idx = np.clip(x_bin_idx, 0, bins - 1)
        y_bin_idx = np.clip(y_bin_idx, 0, bins - 1)
        densities = density[x_bin_idx, y_bin_idx]

        # Normalize density values for color mapping
        densities = densities / densities.max()
        ax.scatter(x, y, s = 2, c = densities, cmap="magma_r")
        # ax.set_xscale("log")
        # ax.set_yscale("log")


    def save_plot(self, fig, type=None, category = "analysis", format = "png", dpi = 300):
        self.image_location = replace_first_folder(self.config.subgraph_location, "Images") + f"_{self.dimension}D"
        print(self.image_location)
        filename = f"{category}_{type}"
        print(f"{filename}.{format}")
        filename = f"{self.image_location}/{filename}"
        fig.savefig(f"{filename}.{format}", format = format, dpi = dpi, transparent = True)
    
    def plot_umis_vs_knn(self, ax = None, additional_arguments = []):
        if not additional_arguments:
            metric_str = f"mean_knn"
        else:
            metric_str = f"mean_morphed_knn"

        print(self.reconstruction_summary[metric_str])
        per_reconstruction_metrics = self.reconstruction_summary.copy().set_index("bc")[metric_str]
        print(per_reconstruction_metrics)
        umis_per_cell = self.base_edges.copy().groupby("cell_bc_10x")["nUMI"].sum()
        print(umis_per_cell)
        print(per_reconstruction_metrics)
        per_reconstruction_metrics_df = per_reconstruction_metrics.to_frame(name=metric_str)

        # 2. Perform a left join with the second Series
        #    .join() needs a DataFrame, so umis_per_cell is converted implicitly
        #    or explicitly (using .to_frame(name='nUMI')).
        #    'how=left' keeps all indices from per_reconstruction_metrics_df.
        combined_df = per_reconstruction_metrics_df.join(
            umis_per_cell.to_frame(name='nUMI'), # Convert second Series here
            how='left'
        )

        print("Combined DataFrame using .join():")
        print(combined_df)

        correlation = combined_df["nUMI"].corr(combined_df[metric_str], method='spearman') # method='pearson' is default

        # 2. Format the correlation text
        #    Displaying with 2 or 3 decimal places is common.
        correlation_text = f"R = {correlation:.2f}"
        ax.text(0.05, 0.95, correlation_text,
            transform=ax.transAxes, # Use axes coordinates
            fontsize=10,
            verticalalignment='top', # Align text box top edge with y coordinate
            horizontalalignment='left',# Align text box left edge with x coordinate
            # Optional: Add a background box for readability
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
        ax.scatter(combined_df["nUMI"], combined_df[metric_str], s = 10)
        ax.set_xscale("log")

    def plot_density_vs_knn(self, ax = None, additional_arguments = []):
        if not additional_arguments:
            metric_str = f"mean_knn"
        else:
            metric_str = f"mean_morphed_knn"
        self.calculate_cell_densities(metric_str= metric_str)
        # filtered_base_edges = self.base_edges[self.base_edges["nUMI"] > 1].copy()
        degree_per_cell = self.base_edges.copy().groupby("cell_bc_10x")["nUMI"].count()
        umis_per_cell = self.base_edges.copy().groupby("cell_bc_10x")["nUMI"].sum()
        self.reconstruction_summary_with_density["nUMI"] = umis_per_cell
        print(self.reconstruction_summary_with_density)
        print(umis_per_cell)
        bc_fused = self.reconstruction_summary_with_density.copy().merge(umis_per_cell.rename('nUMI_10x'), left_on='bc', right_index=True, how='left')
        bc_fused = bc_fused.merge(degree_per_cell.rename('degree'), left_on='bc', right_index=True, how='left')

        ax.scatter(bc_fused["density"], bc_fused[metric_str], s = 2)

        
        correlation = bc_fused["density"].corr(bc_fused[metric_str], method='spearman')**2 # method='pearson' is default

        # 2. Format the correlation text
        #    Displaying with 2 or 3 decimal places is common.
        correlation_text = f"R2 = {correlation:.3f}"
        slope, intercept, r_value, p_value, std_err = linregress(bc_fused["density"], bc_fused["density"])

        # Create the regression line
        line = slope * bc_fused["density"] + intercept

        # Plot the regression line
        # ax.plot(bc_fused["density"], line, color='red', linestyle='-', label=f'Linear Fit (R={r_value:.2f})')

        ax.text(0.05, 0.95, correlation_text,
            transform=ax.transAxes, # Use axes coordinates
            fontsize=10,
            verticalalignment='top', # Align text box top edge with y coordinate
            horizontalalignment='left',# Align text box left edge with x coordinate
            # Optional: Add a background box for readability
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
        

    def calculate_cell_densities(self, metric_str = "mean_knn"):
        df = self.reconstruction_summary
        coords = df[['gt_x', 'gt_y']].dropna().values
        original_indices = df[['gt_x', 'gt_y']].dropna().index # Keep track of original indices

        dataset = coords.T  # Transpose the coordinates

        # 3. Create the gaussian_kde object
        # Using bw_method=None uses Scott's rule for automatic bandwidth estimation.
        # You could also use bw_method='silverman'
        print("Initializing gaussian_kde (using default Scott's rule bandwidth estimation)...")
        try:
            kde = gaussian_kde(dataset, bw_method=None)

            # 4. Evaluate the density *at the original points*
            # Pass the transposed data again to evaluate density at those locations
            # This returns density values directly (not log-density)
            density = kde(dataset)

            # 5. Add density back to the original DataFrame
            df['density'] = np.nan # Initialize column
            df.loc[original_indices, 'density'] = density # Assign using original indices

            print("\nCalculated Gaussian KDE density using scipy.stats.gaussian_kde")
            print(df[['gt_x', 'gt_y', 'density']].head())
            print(f"\nMin density: {df['density'].min():.4g}")
            print(f"Max density: {df['density'].max():.4g}")
            print(f"Mean density: {df['density'].mean():.4g}")
            print(f"NaN density values: {df['density'].isna().sum()}")

        except Exception as e:
            print(f"Error during gaussian_kde calculation: {e}")
            df['density'] = np.nan # Add empty column if calculation fails
        self.reconstruction_summary_with_density = df

    def plot_per_cell_cpd(self, ax=None, additional_arguments = []):
        if not additional_arguments:
            metric_str = f"mean_knn"
        else:
            metric_str = f"mean_morphed_knn"
        if not additional_arguments:
            type = "recon"
        else:
            type = f"morph_recon"
        self.calculate_cell_densities(metric_str=metric_str)
        gt_positions = self.reconstruction_summary_with_density.copy().set_index("bc")
        gt_positions = gt_positions[["gt_x", "gt_y"]].copy()  # Use .copy() to avoid SettingWithCopyWarning
        recon_positions = self.full_reconstruction_summary.copy().set_index("node_bc")
        recon_positions = recon_positions.loc[gt_positions.index, :].copy() # Use .copy()

        recon_columns = recon_positions[[col for col in recon_positions.columns if col.startswith(type)]]

        # Calculate pairwise distances for ground truth positions
        gt_distances_pairwise = squareform(pdist(gt_positions))

        # Initialize a dictionary to store individual point correlations
        individual_correlations = {}

        for bc_index, (gt_x, gt_y) in gt_positions.iterrows():
            print(bc_index)
            # Get the index of the current point in the gt_positions DataFrame
            current_gt_index = gt_positions.index.get_loc(bc_index)

            # Extract the ground truth distances for the current point to all other points
            current_gt_distances = gt_distances_pairwise[current_gt_index, :]

            all_recon_distances_for_point = []

            for i in range(len(recon_columns.columns) // 2):
                recon_x_col = f"{type}_x_{i+1}"
                recon_y_col = f"{type}_y_{i+1}"

                if recon_x_col in recon_columns.columns and recon_y_col in recon_columns.columns:
                    current_recon_positions = recon_columns[[recon_x_col, recon_y_col]]
                    # Ensure the order of rows in recon_positions matches gt_positions
                    current_recon_positions = current_recon_positions.loc[gt_positions.index, :]
                    recon_distances_pairwise = squareform(pdist(current_recon_positions))
                    current_recon_distances = recon_distances_pairwise[current_gt_index, :]
                    all_recon_distances_for_point.append(current_recon_distances)
                else:
                    print(f"Warning: Columns {recon_x_col} or {recon_y_col} not found in recon_columns.")
                    continue

            if all_recon_distances_for_point:
                # Stack the reconstructed distances for all reconstructions for this point
                all_recon_distances_stacked = np.array(all_recon_distances_for_point)

                # Calculate the average reconstructed distances across all reconstructions
                avg_recon_distances = np.mean(all_recon_distances_stacked, axis=0)

                # Calculate the correlation between the ground truth distances and the average reconstructed distances
                if len(current_gt_distances) > 1 and len(avg_recon_distances) > 1:
                    correlation, _ = pearsonr(current_gt_distances, avg_recon_distances)
                    individual_correlations[bc_index] = correlation
                else:
                    individual_correlations[bc_index] = np.nan # Or some other indicator if only one point

            else:
                individual_correlations[bc_index] = np.nan # If no reconstruction columns were processed

        # Convert the dictionary of individual correlations to a Series and add it as a new column to gt_positions
        gt_positions['distance_correlation'] = pd.Series(individual_correlations)
        print(gt_positions)
        pass

    def plot_metric_heatmaps(self, fig=None, additional_arguments = []): 
        if fig:
            plt.close(fig)
            del fig
        if not additional_arguments:
            metric_str = f"mean_knn"
        else:
            metric_str = f"mean_morphed_knn"
        self.calculate_cell_densities(metric_str= metric_str)
        # filtered_base_edges = self.base_edges[self.base_edges["nUMI"] > 1].copy()
        degree_per_cell = self.base_edges.copy().groupby("cell_bc_10x")["nUMI"].count()
        umis_per_cell = self.base_edges.copy().groupby("cell_bc_10x")["nUMI"].sum()
        print(self.reconstruction_summary_with_density)
        print(umis_per_cell)
        bc_fused = self.reconstruction_summary_with_density.copy().merge(umis_per_cell.rename('nUMI'), left_on='bc', right_index=True, how='left')
        bc_fused = bc_fused.merge(degree_per_cell.rename('degree'), left_on='bc', right_index=True, how='left')
        bc_fused["log_nUMI"] = np.log(bc_fused["nUMI"])
        bc_fused["log_degree"] = np.log(bc_fused["degree"])
        print(bc_fused)
        import matplotlib.patches as patches
        import matplotlib.colors as mcolors

        # --- Bin definitions ---
        n_bins_x = 40
        n_bins_y = 40
        if self.sample =="mouse_hippocampus":
            n_bins_x = 20
            n_bins_y = 20

        # Bin the coordinates
        bc_fused['gt_x_bin'] = pd.cut(bc_fused['gt_x'], bins=n_bins_x)
        bc_fused['gt_y_bin'] = pd.cut(bc_fused['gt_y'], bins=n_bins_y)

        # Group by binned coordinates
        grouped = bc_fused.groupby(['gt_x_bin', 'gt_y_bin'])
        points_per_bin = grouped.size().reset_index(name='count')
        median_umi = grouped['log_nUMI'].median().reset_index(name='median_nUMI')
        mean_knn = grouped[metric_str].mean().reset_index(name=metric_str)
        median_degree = grouped["log_degree"].median().reset_index(name='log_degree')



        # Merge data
        plot_data = pd.merge(points_per_bin, median_umi, on=['gt_x_bin', 'gt_y_bin'], how='left')
        plot_data = pd.merge(plot_data, mean_knn, on=['gt_x_bin', 'gt_y_bin'], how='left')
        plot_data = pd.merge(plot_data, median_degree, on=['gt_x_bin', 'gt_y_bin'], how='left')

        # Extract bin boundaries
        plot_data['x_left'] = plot_data['gt_x_bin'].apply(lambda x: x.left)
        plot_data['x_right'] = plot_data['gt_x_bin'].apply(lambda x: x.right)
        plot_data['y_bottom'] = plot_data['gt_y_bin'].apply(lambda x: x.left)
        plot_data['y_top'] = plot_data['gt_y_bin'].apply(lambda x: x.right)

        # --- Plot 1: Count per bin ---
        fig, ax = plt.subplots(figsize=(6, 6))
        norm_count = mcolors.Normalize(vmin=0, vmax=30)
        cmap = plt.cm.magma_r

        for _, row in plot_data.iterrows():
            if row['count'] > 0:
                color = cmap(norm_count(row['count']))
            else:
                color = 'white'  # White for empty bins
            rect = patches.Rectangle(
                (row['x_left'], row['y_bottom']),
                row['x_right'] - row['x_left'],
                row['y_top'] - row['y_bottom'],
                facecolor=color,
                edgecolor='none'
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(norm=norm_count, cmap=cmap)
        ax.scatter(bc_fused['gt_x'], bc_fused['gt_y'],s =0)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Number of Cells")

        ax.set_title("Cell Density (per bin)")
        ax.set_xlabel("gt_x")
        ax.set_ylabel("gt_y")
        ax.set_aspect("equal")
        plt.tight_layout()
        img_format = "pdf"
        fig.savefig(f"Images/{self.sample}_density.{img_format}", dpi=300, format = img_format)

        # --- Plot 2: Median nUMI ---
        fig, ax = plt.subplots(figsize=(6, 6))
        norm_umi = mcolors.Normalize(vmin=3, vmax=9) # plot_data['median_nUMI'].max()

        for _, row in plot_data.iterrows():
            if not np.isnan(row['median_nUMI']):
                color = cmap(norm_umi(row['median_nUMI']))
            else:
                color = 'white'
            rect = patches.Rectangle(
                (row['x_left'], row['y_bottom']),
                row['x_right'] - row['x_left'],
                row['y_top'] - row['y_bottom'],
                facecolor=color,
                edgecolor='none'
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(norm=norm_umi, cmap=cmap)
        sm.set_array([])
        ax.scatter(bc_fused['gt_x'], bc_fused['gt_y'],s =0)
        plt.colorbar(sm, ax=ax, label="Median nUMI")

        ax.set_title("Median nUMI per Bin")
        ax.set_xlabel("gt_x")
        ax.set_ylabel("gt_y")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(f"Images/{self.sample}_median_log_umi.{img_format}", dpi=300, format = img_format)

        # --- Plot 3: mean KNN ---
        fig, ax = plt.subplots(figsize=(6, 6))
        norm_umi = mcolors.Normalize(vmin=0, vmax=1)

        for _, row in plot_data.iterrows():
            if not np.isnan(row[metric_str]):
                color = cmap(norm_umi(row[metric_str]))
            else:
                color = 'white'
            rect = patches.Rectangle(
                (row['x_left'], row['y_bottom']),
                row['x_right'] - row['x_left'],
                row['y_top'] - row['y_bottom'],
                facecolor=color,
                edgecolor='none'
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(norm=norm_umi, cmap=cmap)
        sm.set_array([])
        ax.scatter(bc_fused['gt_x'], bc_fused['gt_y'],s =0)
        plt.colorbar(sm, ax=ax, label="Mean KNN")

        ax.set_title("Mean KNN per Bin")
        ax.set_xlabel("gt_x")
        ax.set_ylabel("gt_y")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(f"Images/{self.sample}_mean_knn.{img_format}", dpi=300, format = img_format)

        # --- Plot 4: median degree ---
        fig, ax = plt.subplots(figsize=(6, 6))
        norm_umi = mcolors.Normalize(vmin=2, vmax=8)

        for _, row in plot_data.iterrows():
            if not np.isnan(row['log_degree']):
                color = cmap(norm_umi(row['log_degree']))
            else:
                color = 'white'
            rect = patches.Rectangle(
                (row['x_left'], row['y_bottom']),
                row['x_right'] - row['x_left'],
                row['y_top'] - row['y_bottom'],
                facecolor=color,
                edgecolor='none'
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(norm=norm_umi, cmap=cmap)
        sm.set_array([])
        ax.scatter(bc_fused['gt_x'], bc_fused['gt_y'],s =0)
        plt.colorbar(sm, ax=ax, label="Median degree")

        ax.set_title("Median degree per Bin")
        ax.set_xlabel("gt_x")
        ax.set_ylabel("gt_y")
        ax.set_aspect("equal")
        plt.tight_layout()
        fig.savefig(f"Images/{self.sample}_median_degree.{img_format}", dpi=300, format = img_format)

        correlation_data = plot_data[["count", "median_nUMI", "log_degree", metric_str]].copy()
        correlation_matrix = correlation_data.corr(method='spearman')**2  # or 'spearman'

        # --- Plot the correlation matrix as a heatmap ---
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Oranges", cbar=True, ax=ax)

        ax.set_title("Correlation Between Binned Features")
        plt.tight_layout()
        fig.savefig(f"Images/{self.sample}_correlations.{img_format}", dpi=300, format = img_format)

        pass
class gatedSubgraphCollection():
    def __init__(self, list_of_gated_subgraphs, config):
        self.all_subgraphs: List[gatedSubgraph] = list_of_gated_subgraphs
        self.n_subgraphs:int = len(self.all_subgraphs)
        self.config = config
        self.unmodified_subgraph:gatedSubgraph = None

    def __iter__(self):
        for subgraph in self.all_subgraphs:
            yield subgraph

    def plot_all_distance_distributions(self):
        fig, axes = plt.subplots(1, self.n_subgraphs, figsize = (6*self.n_subgraphs, 6))
        if self.n_subgraphs ==1:
            axes = [axes]

        for i, subgraph in enumerate(self):
            c_ax = axes[i]
            subgraph.plot_distance_distribution(ax = c_ax)
            c_ax.set_title(f"{subgraph.gating_threshold}")

    def plot_all_subgraphs(self, plotting_type = None, additional_metrics = []):
        # fig, axes = plt.subplots(1, self.n_subgraphs, figsize = (6*self.n_subgraphs, 6))
        # if self.n_subgraphs ==1:
        #     axes = [axes]
        dual_plot_types = ["normal_vs_morphed"]
        save_format = self.config.vizualisation_args.save_to_image_format
        n_subgraphs = self.n_subgraphs
        max_columns = 6
        if n_subgraphs<max_columns:
            max_columns =n_subgraphs
        else:
            max_columns = 6
        n_rows = (n_subgraphs + max_columns - 1) // max_columns  # Calculate the required rows
        
        fig, axes = plt.subplots(n_rows, max_columns, figsize=(max_columns * 6, n_rows * 6))
        axes = np.array(axes)  # Ensure it's an array for indexing
        axes = axes.flatten()  # Flatten to handle single indexing
        
        for i, subgraph in enumerate(self):
            ax1 = axes[i]
            if plotting_type in dual_plot_types:
                divider = ax1.get_position()  # Get original axis position
                fig.delaxes(ax1)  # Remove original axis from subplot grid
                gap = 0.1  # Add vertical spacing
                height = (divider.height / 2) - gap  # Reduce height for spacing

                ax1 = fig.add_axes([divider.x0, divider.y0 + height + gap, divider.width, height])  # Upper axis
                ax2 = fig.add_axes([divider.x0, divider.y0, divider.width, height])  # Lower axis
            else:
                ax1.set_title(f"{subgraph.gating_threshold}, {len(subgraph.edgelist)} edges")
                ax1.set_box_aspect(1)

            if plotting_type == "distance_variation_metrics":
                subgraph.plot_no_gt_metric_correlation(ax=ax1)
            elif plotting_type == "distance_distribution":
                subgraph.plot_distance_distribution(ax=ax1)
            elif plotting_type == "distance_distribution_std":
                subgraph.plot_distance_distribution(ax=ax1, data_type="std")
            elif plotting_type in ["distortion", "knn", "edge_distance"]: # edge distance here is the per cella verage edge distance
                subgraph.plot_gt_reconstruction_metric_distribution(ax=ax1, metric=plotting_type, additional_arguments = additional_metrics)
            elif plotting_type == "knn_over_k":
                subgraph.plot_knn_over_k(ax = ax1, additional_arguments=additional_metrics)
            elif plotting_type == "metric_correlation":
                subgraph.plot_reconstruction_metric_correlation(ax=ax1, additional_arguments = additional_metrics)
            elif plotting_type == "metric_correlation_std":
                subgraph.plot_reconstruction_metric_correlation(ax=ax1, additional_arguments = additional_metrics, type="std")
            elif plotting_type == "dbscan_calculation":
                subgraph.calculate_recon_dbscan_clusters(additional_arguments=additional_metrics)
                subgraph.save_modified_full_reconstruction_summary()
            elif plotting_type == "dbscan_analysis":
                fig = subgraph.analyse_dbscan_clusters(ms=additional_metrics)
            #dual plotting type plots
            elif plotting_type == "normal_vs_morphed":
                subgraph.compare_normal_and_morphed(ax_normal=ax1, ax_morph=ax2)
                subgraph.compare_normal_and_morphed(ax_normal=ax1, ax_morph=ax2, type = "scatter")
            elif plotting_type =="cell_types":
                subgraph.per_cell_type_metrics(fig = fig, additional_arguments = additional_metrics) 
            elif plotting_type =="cpd":
                subgraph.plot_cpd_scatter(ax = ax1, additional_arguments = additional_metrics)
                save_format = "png"
                if additional_metrics and i == self.n_subgraphs-1:
                    plotting_type = plotting_type + "_" + str(additional_metrics[0])
            elif plotting_type == "umi_vs_dist":
                subgraph.plot_recon_dist_vs_umis(ax = ax1)
            elif plotting_type =="knn_over_density":
                subgraph.plot_density_vs_knn(ax = ax1, additional_arguments = additional_metrics)
            elif plotting_type =="knn_over_umis":
                subgraph.plot_umis_vs_knn(ax = ax1, additional_arguments = additional_metrics)
            elif plotting_type =="per_cell_cpd":
                subgraph.plot_per_cell_cpd(ax = ax1, additional_arguments = additional_metrics)
            elif plotting_type =="metric_heatmaps":
                subgraph.plot_metric_heatmaps(fig = fig, additional_arguments = additional_metrics)
            else: 
                print("No such analysis:", plotting_type)
                
                return
        
        # Hide any unused subplot axes
        for j in range(n_subgraphs, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"{plotting_type}")
        # fig.tight_layout()
        # plt.tight_layout()
        subgraph.save_plot(fig, type = plotting_type, category = "analysis", format = save_format)
    
    def plot_positions(self, plotting_type = None, additional_metrics =[]):
        n_subgraphs = self.n_subgraphs
        save_format = self.config.vizualisation_args.save_to_image_format
        max_columns = 6
        figsize = 12
        dual_plot_types = ["gt_uni_edges"]
        if n_subgraphs<max_columns:
            max_columns =n_subgraphs
        else:
            max_columns = 6
        n_rows = (n_subgraphs + max_columns - 1) // max_columns  # Calculate the required rows
        
        fig, axes = plt.subplots(n_rows, max_columns, figsize=(max_columns * figsize, n_rows * figsize))
        if plotting_type in dual_plot_types:
            fig2, axes2 = plt.subplots(n_rows, max_columns, figsize=(max_columns * figsize, n_rows * figsize))
            axes2 = np.array(axes2)  # Ensure it's an array for indexing
            axes2 = axes2.flatten()  # Flatten to handle single indexing
        axes = np.array(axes)  # Ensure it's an array for indexing
        axes = axes.flatten()  # Flatten to handle single indexing
        
        for i, subgraph in enumerate(self):
            ax1 = axes[i]
            if plotting_type in dual_plot_types:
                ax2 = axes2[i]
                ax2.set_title(f"{subgraph.gating_threshold}, {len(subgraph.edgelist)} edges")
                ax2.set_box_aspect(1)
                ax2.set_title(f"{subgraph.gating_threshold}")
            ax1.set_title(f"{subgraph.gating_threshold}, {len(subgraph.edgelist)} edges")
            ax1.set_box_aspect(1)
            ax1.set_title(f"{subgraph.gating_threshold}")

            if plotting_type == "beads":
                subgraph.plot_only_beads(ax = ax1) 
            elif plotting_type == "gt_uni_edges":
                subgraph.plot_gt_edges(ax = ax1, ax_hist = ax2, type = plotting_type, additional_arguments=additional_metrics) 
            elif plotting_type =="edges_beads_est":
                subgraph.plot_edges_gt_estimated_beads(ax = ax1)
            elif plotting_type == "cell_types":
                subgraph.plot_cell_types(fig = fig, additional_arguments = additional_metrics, subgraph_collection=self)
            elif plotting_type =="node_types":
                subgraph.plot_node_types(fig = fig, additional_arguments = additional_metrics)
                print("No such analysis:", plotting_type)
            
        
        # Hide any unused subplot axes
        for j in range(n_subgraphs, len(axes)):
            axes[j].set_visible(False)
        # plt.show( )
        fig.suptitle(f"{plotting_type}")
        if plotting_type in dual_plot_types:
            fig2.suptitle(f"{plotting_type}")
            subgraph.save_plot(fig2, type = plotting_type+"_2", category = "positions", format = "png", dpi = 900)
        subgraph.save_plot(fig, type = plotting_type, category = "positions", format = save_format, dpi = 600)

def load_quality_files(config):
 
    config.gated_file_location = replace_first_folder(config.subgraph_location, "Output_files") + f"_{config.filter_analysis_args.reconstruction_dimension}D"
    if config.subgraph_to_analyse.gating_threshold !="all":
        subgraphs = [file for file in os.listdir(config.gated_file_location) if "detailed_edgelist" in file and ".csv" in file and (f"_{config.subgraph_to_analyse.gating_threshold}_" in file or ("gated" not in file and "dbscan" not in file))]
    else:
        subgraphs = [file for file in os.listdir(config.gated_file_location) if "detailed_edgelist" in file and ".csv" in file and (config.modification_type in file or ("gated" not in file and "dbscan" not in file))]
    unmodded_subgraph = [file for file in subgraphs if len(re.findall(f"_{config.modification_type}_",file))==0]
    if not config.subgraph_to_analyse.include_recursively_gated:
        subgraphs = [file for file in subgraphs if len(re.findall(f"_{config.modification_type}_",file))<2]

    if not config.subgraph_to_analyse.include_ungated:
        subgraphs = [file for file in subgraphs if len(re.findall(f"_{config.modification_type}_",file))!=0]

    if not subgraphs:
        raise FileNotFoundError(f"No files fulfill requirements - gating_threshold:{config.subgraph_to_analyse.gating_threshold}, include_ungated: {config.subgraph_to_analyse.include_ungated}, include_recursively_gated:{config.subgraph_to_analyse.include_recursively_gated}")
    subgraphs = [gatedSubgraph(config, file) for file in subgraphs]
    all_subgraphs = gatedSubgraphCollection(subgraphs, config)
    all_subgraphs.unmodified_subgraph = gatedSubgraph(config, unmodded_subgraph[0])
    return all_subgraphs


def additional_subgraph_analysis(config, category =None, plotting_type = None, additional_arguments = []):
    from subgraph_analysis_functions import initalize_files
    
    config = initalize_files(config)
    config = initialize_post_subgraph_analysis(config, initial=True)
    if config.filter_analysis_args.analyse_all_thresholds:
        all_thresholds_with_reconstruction =[int(re.search(fr"{config.filter_analysis_args.filter}_(\d+)", filter).group(1)) for filter in os.listdir(replace_first_folder(config.subgraph_base_location, "Subgraph_reconstructions")) if config.filter_analysis_args.filter in filter and f"{config.filter_analysis_args.reconstruction_dimension}D" in filter]
    else:
        all_thresholds_with_reconstruction = config.filter_analysis_args.thresholds_to_analyse
    for threshold in all_thresholds_with_reconstruction:
        config.subgraph_to_analyse.threshold = threshold
        print("\nCurrent Threshold: ",threshold)
        config = initialize_post_subgraph_analysis(config)
        config.vizualisation_args.show_plots = False
        config.plot_modification = True

        all_subgraphs = load_quality_files(config)
        if category == "analyse": 
            all_subgraphs.plot_all_subgraphs(plotting_type=plotting_type, additional_metrics=additional_arguments)
        elif category == "positions":
            # all_subgraphs.plot_positions(plotting_type = "beads")
            all_subgraphs.plot_positions(plotting_type = plotting_type, additional_metrics = additional_arguments)
        else:
            print("No such category:", category)
        # print(subgraph.quality_df)
        # print(subgraph.all_enriched_subgraphs)
   
    return

if __name__== "__main__":

    from Utils import *
    # config_subgraph_analysis_mouse_embryo, config_subgraph_analysis_mouse_hippocampus, config_subgraph_analysis_tonsil, config_subgraph_analysis_mouse_embryo_uni
    # config = ConfigLoader('config_subgraph_analysis_tonsil.py')
    # config_analysis
    config = ConfigLoader('config_subgraph_analysis_tonsil.py')
    # config.vizualisation_args.save_to_image_format = "pdf" 
    categories = ["analyse"] # analyse or position
    plot_what = ["dbscan_calculation", "dbscan_analysis"]
    # standard set of runs
    # categories = ["analyse"]
    # plot_what = ["dbscan_calculation", "dbscan_analysis"]
    # plotting_type = "gt_uni_edges"
    additional_arguments = [range(5, 16), range(2,8)]
    for cat in categories:
        for plotting_type in plot_what:
            additional_subgraph_analysis(config, category = cat, plotting_type = plotting_type, additional_arguments = additional_arguments)
    plt.show()