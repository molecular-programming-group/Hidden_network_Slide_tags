import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils import *
import re
import seaborn as sns

class groupedSamples():
    def __init__(self, all_samples, all_gt_positions):
        self.samples = [analysisSample(sample, gt_positions) for sample, gt_positions in zip(all_samples, all_gt_positions)]
        self.load_sample_parameters()
        self.load_colors()

    def load_colors(self):
        np.random.seed(42)
        num_colors_needed = len(self)
        cmap = plt.cm.plasma
        self.colors = cmap(np.random.choice(np.arange(cmap.N), num_colors_needed, replace=False))
        self.colors = [(243/255,147/255,0/255,1),(137/255, 75/255, 162/255, 1), (0/255,166/255,187/255,1)]
        
    def load_sample_parameters(self):
        n_cells, n_beads, n_edges, n_umis, self.names = [], [], [], [], []
        n_cells_gt, n_beads_gt, n_edges_gt, n_umis_gt = [], [], [], []
        for sample in self:
            self.names.append(sample.name)
            n_cells.append(sample.n_cells)
            n_beads.append(sample.n_beads)
            n_edges.append(sample.n_edges)
            n_umis.append(sample.n_umis)
            n_cells_gt.append(sample.n_cells_gt)
            n_beads_gt.append(sample.n_beads_gt)
            n_edges_gt.append(sample.n_edges_gt)
            n_umis_gt.append(sample.n_umis_gt)
        self.n_cells, self.n_beads, self.n_edges, self.n_umis = n_cells, n_beads, n_edges, n_umis
        self.n_cells_gt, self.n_beads_gt, self.n_edges_gt, self.n_umis_gt = n_cells_gt, n_beads_gt, n_edges_gt, n_umis_gt
        self.load_colors()
    
    def __iter__(self):
        for sample in self.samples:
            yield sample

    def __len__(self):
        return len(self.samples)
    
    def find_lowest(self, target_type = None):
        if not target_type:
            target_type = "nUMI"

    def plot_unipartite_edgelength_histogram(self, format = "png"):
        for sample in self:
            sample.load_cell_unipartite()
            print(sample.n_cells)
            total_number_of_distances = (sample.n_cells*(sample.n_cells-1))/2
            fig, ax_hist = plt.subplots(figsize = (6,6))

            
            unipartite_df = sample.unipartite_cell
            # print(unipartite_df)
            # print(sample.gt_df)

            only_gt_edges = unipartite_df[(unipartite_df["source"].isin(sample.gt_df.index)) & (unipartite_df["target"].isin(sample.gt_df.index))]
            x_span = sample.gt_df["x"].max() - sample.gt_df["x"].min()
            y_span = sample.gt_df["y"].max() - sample.gt_df["y"].min()
            print(y_span, x_span)
            continue
            cells_with_edge = list(set(only_gt_edges["source"]).union(set(only_gt_edges["target"])))
            positions_cells_with_edges = sample.gt_df.loc[cells_with_edge, :]

            source_positions = positions_cells_with_edges.loc[only_gt_edges["source"], ["x", "y"]].values
            target_positions = positions_cells_with_edges.loc[only_gt_edges["target"], ["x", "y"]].values
            edge_lengths = np.linalg.norm(source_positions - target_positions, axis=1)

            from scipy.spatial.distance import pdist
            all_cell_positions = positions_cells_with_edges.loc[:,["x", "y"]].values
            all_possible_distance = pdist(all_cell_positions)
            max_bin_value = max(all_possible_distance.max(), edge_lengths.max())

            # Define consistent bin edges (same for both histograms)
            num_bins = 300
            bin_edges = np.linspace(0, max_bin_value, num_bins + 1)
            counts_all, bin_edges, patches = ax_hist.hist(all_possible_distance, bins=bin_edges, alpha = 0.5, facecolor = "lightgray")
            # Compute histogram bins and colors
            counts_edges, bin_edges, patches = ax_hist.hist(edge_lengths, bins=bin_edges, alpha=1)

            # ax_ratio = ax_hist.twinx()
            fig, ax_ratio = plt.subplots(figsize = (6,6))
            ax_ratio.set_box_aspect(1)
            
            hist_ratio = np.divide(counts_edges, counts_all, where=counts_all > 0)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins

            # ax_ratio.plot(bin_centers, hist_ratio, color="red", linewidth=2, label="Ratio")
            ax_ratio.scatter(bin_centers, hist_ratio, color="red", s=2, label="Ratio")
            ax_ratio.set_ylim([0,1.05])
            # ax_ratio.set_xlim([-50, 1000]) 
            ax_ratio.set_box_aspect(1)
            ax_ratio_hist = ax_hist.twinx()
            ax_ratio_hist.scatter(bin_centers, hist_ratio, color="red", s=2, label="Ratio")
            ax_ratio_hist.set_ylim([0, 1.1])
            ax_ratio_hist.axhline(len(only_gt_edges)/len(all_possible_distance), c = "r", linestyle = "--")
            ax_ratio_hist.axhline(len(unipartite_df)/total_number_of_distances, c = "m", linestyle = "--")

            total_number_of_distances
            # Labels and titles for histogram
            ax_hist.set_xlabel("Edge Length") 
            ax_hist.set_ylabel("Frequency")
            ax_hist.set_title(f"total edges: {len(unipartite_df)}, gt: {len(only_gt_edges)}")

            # ax_hist.set_xlim([-10, bin_edges[-1]*1.05])
            ax_hist.set_box_aspect(1)
            plt.title(f"{len(unipartite_df)}/{total_number_of_distances} edges vs complete")
            fig.savefig(f"Images/gt_uni_distribution_{sample.name}.{format}", format = format)
        
    def plot_distributions_beads(self, format = "png"):
        for subgraph in self:
            edges = subgraph.edges
            umis_sum = edges.groupby("bead_bc")["nUMI"].sum().reset_index()
            total_umis = edges["nUMI"].sum()
            umis_sum.rename(columns={"nUMI": "umis_sum"}, inplace=True)

            bead_degree = edges["bead_bc"].value_counts().reset_index()
            bead_degree.columns = ["bead_bc", "degree"]

            bead_stats_base = pd.merge(umis_sum, bead_degree, on="bead_bc")
            bead_stats_base["ratio"] = bead_stats_base["umis_sum"]/bead_stats_base["degree"]
            
            bead_stats = bead_stats_base[(bead_stats_base["ratio"]<10)]
            sorted_umis = np.sort(bead_stats_base["umis_sum"].unique())
            for umi_threshold in sorted_umis:
                failed_below_threshold = ((bead_stats_base["ratio"] < 10) & (bead_stats_base["umis_sum"] > umi_threshold)).sum()
                passed_above_threshold = ((bead_stats_base["ratio"] >= 10) & (bead_stats_base["umis_sum"] > umi_threshold)).sum()
                print(failed_below_threshold, passed_above_threshold)
                # Stop when failing points become >= passing points
                if failed_below_threshold <= passed_above_threshold:
                    final_umi_threshold = umi_threshold
                    break
                else:
                    final_umi_threshold = 1
            if subgraph.name == "tonsil":
                final_umi_threshold =1500
            elif subgraph.name =="mouse_embryo":
                final_umi_threshold =800
            elif subgraph.name =="mouse_hippocampus":
                final_umi_threshold =500

            # print(bead_stats)
            # print(counted_umis)¨
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize= (16, 6), layout = "tight")
            plt.suptitle(f"{subgraph.name}, {subgraph.n_beads} beads")
            umis_counts = np.unique(bead_stats_base["umis_sum"], return_counts=True)
            ax1.scatter(umis_counts[0], umis_counts[1], s = 5, alpha = 0.5)
            ax1.set_xlabel("Total UMIs")
            ax1.set_ylabel("Count")
            ax1.set_yscale("log")
            ax1.set_xscale("log")
            ax1.set_title(f"UMIS sum Distribution")
            ax1.axvline(final_umi_threshold, c = "r", linestyle = "--", label = f"{final_umi_threshold}")
            fig.savefig(f"Images/bead_distributions_{subgraph.name}.{format}", format = format)
            continue

            degree_counts = np.unique(bead_stats_base["degree"], return_counts=True)
            ax3.scatter(degree_counts[0], degree_counts[1], s = 5, alpha = 0.5)
            ax3.set_xlabel("Degree")
            ax3.set_ylabel("Count")
            ax3.set_yscale("log")
            ax3.set_xscale("log")
            ax3.set_title("Degree_distribution")

            ax2.scatter(bead_stats["degree"], bead_stats["umis_sum"], s = bead_stats["ratio"], alpha = 0.5)
            ax2.set_xlabel("Total UMIs")
            ax2.set_ylabel("Bead Degree")
            ax2.axvline(1500)
            x = np.linspace(bead_stats["degree"].min(), bead_stats["degree"].max(), 100)
            ax2.plot(x, x*10)
            ax2.axhline(final_umi_threshold, c = "r", linestyle = "--", label = f"{final_umi_threshold}")

            ax2.set_title(f"UMI sum over degree {final_umi_threshold}")
            from scipy.stats import pearsonr, linregress
            correlation, p_value = pearsonr(bead_stats["degree"], bead_stats["umis_sum"])
            print(f"Pearson Correlation: {correlation:.2f}, P-value: {p_value:.2e}")
            slope, intercept, _, _, _ = linregress(bead_stats["degree"], bead_stats["umis_sum"])
            x_fit = np.linspace(1, bead_stats["degree"].max(), 100)
            y_fit = slope * x_fit + intercept
            ax2.plot(x_fit, y_fit, color="red", linestyle="--", label=f"Correlation ratio <{10}: {correlation:.2f}")
            # ax2.set_yscale("log")
            # ax2.set_xscale("log")
            ax2.legend(loc="upper left")

            bead_stats = bead_stats_base[bead_stats_base["ratio"]>=10][bead_stats_base["degree"]>=1]
            ax2.scatter(bead_stats["degree"], bead_stats["umis_sum"], s = bead_stats["ratio"], alpha = 0.5, c = "r")
            ax2.set_xlabel("Bead Degree")
            ax2.set_ylabel("Total UMIs")

            from scipy.stats import pearsonr, linregress
            correlation, p_value = pearsonr(bead_stats_base["degree"], bead_stats_base["umis_sum"])
            print(f"Pearson Correlation: {correlation:.2f}, P-value: {p_value:.2e}")
            slope, intercept, _, _, _ = linregress(bead_stats_base["degree"], bead_stats_base["umis_sum"])
            x_fit = np.linspace(1, bead_stats_base["degree"].max(), 100)
            y_fit = slope * x_fit + intercept
            ax2.plot(x_fit, y_fit, color="red", linestyle="--", label=f"Correlation all points: {correlation:.2f}")
            ax2.set_yscale("log")
            ax2.set_xscale("log")
            ax2.legend(loc="upper left")
            fig.savefig(f"Images/bead_distributions_{subgraph.name}.{format}", format = format)

    def plot_degree_distributions(self, format = "png"):
        fig, axes = plt.subplots(1, 2, figsize = (12, 6))
        # colors = ["b", "g", "r", "m", "y", "k", "c"]
        from scipy.stats import lognorm
        colors = self.colors
        for i, sample in enumerate(self):
            ax_cells = axes[0]
            ax_beads = axes[1]
            ax_cells.scatter(sample.cell_degree_dist[0], sample.cell_degree_dist[1], s = 4, label = f"{sample.name} N={sample.cell_degree_dist[1].sum()}", color = colors[i])
            ax_cells.axvline(sample.mean_cell_degrees, label = f"mean {sample.mean_cell_degrees:.3f} ±{sample.std_cell_degrees:.3f}", color=colors[i])
            ax_beads.scatter(sample.bead_degree_dist[0], sample.bead_degree_dist[1], s = 4, label = f"{sample.name} N={sample.bead_degree_dist[1].sum()}", color = colors[i])
            ax_beads.axvline(sample.mean_bead_degrees, label = f"mean degree: {sample.mean_bead_degrees}", color=colors[i])
            
            cell_degrees = sample.cell_degrees[1]  # <- Make sure this exists as a list or array of degree values
            # Remove zeros (lognorm requires positive values)
            cell_degrees = np.array(cell_degrees)

            # Fit the lognormal distribution
            shape, loc, scale = lognorm.fit(cell_degrees, floc=0)  # Force location to 0 for typical lognorm

            # Get mean and std of the underlying normal distribution
            mu = np.log(scale)
            sigma = shape

            # Plot the degree histogram or KDE if needed (optional)
            # sns.histplot(cell_degrees, bins=50, stat="density", kde=True, ax=ax_cells)

            # Plot the PDF on top
            x_vals = np.linspace(min(cell_degrees), max(cell_degrees), 5000)
            pdf_vals = lognorm.pdf(x_vals, shape, loc=loc, scale=scale)*len(cell_degrees)
            x_vals = x_vals[pdf_vals>=1]
            pdf_vals = pdf_vals[pdf_vals>=1]
            # ax_cells.plot(x_vals, pdf_vals, linestyle='--', color=colors[i], label=f"lognorm fit\nμ={mu:.2f}, σ={sigma:.2f}")


        ax_cells.set_ylabel("Count")
        ax_cells.set_xlabel("Degree")
        ax_cells.set_xscale("log")
        ax_cells.set_yscale("log")
        ax_cells.legend(fontsize = 6)
        ax_cells.set_title("Cells")
        ax_cells.set_box_aspect(1)

        ax_beads.set_ylabel("Count")
        ax_beads.set_xlabel("Degree")
        ax_beads.set_xscale("log")
        ax_beads.set_yscale("log")
        ax_beads.legend(fontsize = 6)
        ax_beads.set_title("Beads")
        ax_beads.set_box_aspect(1)

        fig.savefig(f"Images/degree_distributions.{format}", format = format)

    def plot_degrees_per_cell_type(self, format="png"):
        for sample in self:
            print(sample.cell_degrees)
            print(sample.cell_degrees[0])
            print(sample.gt_df)
            degree_df = pd.DataFrame(sample.cell_degrees[1], index=sample.cell_degrees[0], columns=["degree"])
            gt_with_degree = sample.gt_df.copy().join(degree_df)
            print(degree_df)
            print(gt_with_degree)
            
            df = gt_with_degree.reset_index()
            median_order = df.groupby("cell_type")["degree"].median().sort_values(ascending=False).index.tolist()

            # Set up the plot
            plt.figure(figsize=(10, 6))
            sns.violinplot(
                data=df,
                x="cell_type",
                y="degree",
                order=median_order,
                inner="box",
                cut=0
            )
            # Improve layout
            plt.xticks(rotation=45)
            plt.xlabel("Cell Type")
            plt.ylabel("Degree")
            plt.title(f"Degree Distribution per Cell Type {sample.name}")
            plt.tight_layout()
            # plt.show()
        
    
    def plot_counts(self, format = "png"):
        # parameters = [self.n_cells, self.n_beads, self.n_edges, self.n_umis, self.viable_reads]
        parameters_gt = [self.n_cells_gt, self.n_edges_gt, self.n_beads_gt, self.n_umis_gt, self.total_reads, self.viable_reads]
        parameters_total = [self.n_cells, self.n_edges, self.n_beads, self.n_umis, self.total_reads, self.viable_reads]

        titles = ["Total number of cell", "Total number of edges", "Total bead count", "Total umi count", "Total read count", "Viable reads"]
        fig, axes = plt.subplots(1, len(parameters_gt), figsize = (18, 6))

        # colors = ["b", "g", "r", "m", "y", "k", "c"]
        # colors = colors[:len(self)]
        colors = self.colors

        for ax, values_all, values_gt, title in zip(axes, parameters_total, parameters_gt, titles):
            if len(self.names) != len(values_all):
                values_all = values_all[:len(self.names)]
            ax.bar(self.names, values_all, color = colors)
            print(values_gt)
            if values_gt[0]:
                if len(self.names) != len(values_gt):
                    values_gt = values_gt[:len(self.names)-1]
                ax.bar(self.names, values_gt, color = "gray", alpha = 0.5)
            ax.set_box_aspect(1)
            ax.set_title(title)
            ax.set_ylabel("Count")
            ymin, ymax = ax.get_ylim()

            # Increase upper limit by 10%
            ax.set_ylim(ymin, ymax * 1.05)
            ax.set_xticks(range(len(self.names)))  # Set correct number of ticks
            ax.set_xticklabels(self.names, rotation=50)  # Apply labels with rotation
            for p in ax.patches:
                ax.text(
                    p.get_x() + p.get_width() / 2,  # X-position (center of the bar)
                    p.get_height() + 0.5,           # Y-position (slightly above the bar)
                    f'{p.get_height():,.0f}',        # Text (rounded bar height)
                    ha='center', va='bottom',       # Center alignment
                    fontsize=10
                )
        plt.tight_layout()
        fig.savefig(f"Images/sample_properties.{format}", format = format)

    def generate_bead_downsampled_samples(self, target_sample = None, fraction = 100):
        if not target_sample:
            # target_sample = self.find_lowest(target_type)
            target_sample = "mouse_embryo"

        for i, sample in enumerate(self):
            # print(sample.edges)
            if sample.name == target_sample:
                target_number = len(sample.edges["bead_bc"].unique())

        for sample in self:
            if sample.modified:
                continue
            # print(sample.edges["nUMI"])
            print(sample.edges)
            numbers = sample.edges["bead_bc"].unique()

            total_number = len(numbers)
            diff = total_number-target_number
            print(diff, sample.name)
            if diff <1:
                continue
            diff = round(diff*fraction/100)
            np.random.seed(42)
            modified_edges = sample.edges.copy()
            bead_bc_to_remove = np.random.choice(numbers, diff, replace=False)

            modified_edges = modified_edges[~modified_edges["bead_bc"].isin(bead_bc_to_remove)]

            modified_edges.to_csv(f"Intermediary_files/{sample.name}/modified_beads_{fraction}.csv", index = False)
            modified_sample = analysisSample(sample = sample.name, true_name = f"{sample.name}_modified_beads_{fraction}", gt_positions=sample.gt_positions_file, specific_edgelist=f"modified_beads_{fraction}.csv")
            self.samples.append(modified_sample)
            self.load_sample_parameters()

    def generate_edges_downsampled_samples(self, target_sample = None, fraction = 100):
        if not target_sample:
            # target_sample = self.find_lowest(target_type)
            target_sample = "mouse_embryo"

        for i, sample in enumerate(self):
            # print(sample.edges)
            if sample.name == target_sample:
                target_number = len(sample.edges)
        for sample in self:
            if sample.modified:
                continue
            # print(sample.edges["nUMI"])
            numbers = sample.edges
            total_number = len(numbers)
            diff = total_number-target_number
            print(diff, sample.name)
            print(total_number, target_number, diff, "oöiasdf")
            if diff <1:
                continue
            diff = round(diff*fraction/100)
            np.random.seed(42)
            idx = np.random.choice(total_number, diff, replace=False)
            print(len(idx), diff, "#huh")
            print(idx)
            modified_edges = sample.edges.copy()
            modified_edges.drop(idx, inplace=True)
            ensure_directory(f"Intermediary_files/{sample.name}")
            modified_edges.to_csv(f"Intermediary_files/{sample.name}/modified_edges_{fraction}.csv", index = False)
            modified_sample = analysisSample(sample = sample.name, true_name = f"{sample.name}_modified_edges_{fraction}", gt_positions=sample.gt_positions_file, specific_edgelist=f"modified_edges_{fraction}.csv")
            self.samples.append(modified_sample)
            self.load_sample_parameters()

    def generate_nUMI_downsampled_samples(self, target_sample = None, fraction = 100):
        if not target_sample:
            # target_sample = self.find_lowest(target_type)
            target_sample = "mouse_embryo"

        for i, sample in enumerate(self):
            # print(sample.edges)
            if sample.name == target_sample:
                target_number = sample.edges["nUMI"].sum()
        
        for sample in self:
            if sample.modified:
                continue
            # print(sample.edges["nUMI"])
            numbers = sample.edges["nUMI"].values
            total_number = sum(numbers)
            diff = total_number-target_number
            print(diff, sample.name)
            if diff <1:
                continue
            index_pool = np.repeat(np.arange(len(numbers)), numbers)
            diff = round(diff*fraction/100)
            print(index_pool)
            for i in range(diff):
                reduced = False
                while not reduced:
                    idx = np.random.choice(index_pool)  # Random index
                    if numbers[idx] >0:
                        numbers[idx] -= 1  # Reduce by 1
                        reduced = True
                    # print(i)

            modified_edges = sample.edges.copy()
            modified_edges["nUMI"] = numbers
            reduced_edges = modified_edges.loc[modified_edges["nUMI"]!=0, :]
            reduced_edges.to_csv(f"Intermediary_files/{sample.name}/modified_nUMI_{fraction}.csv", index = False)
            modified_sample = analysisSample(sample = sample.name, true_name = f"{sample.name}_modified_nUMI_{fraction}", gt_positions=sample.gt_positions_file, specific_edgelist=f"modified_nUMI_{fraction}.csv")
            self.samples.append(modified_sample)
            self.load_sample_parameters()

        pass
    def load_modified_samples(self):
        import os
        for sample in self:
            if sample.name == "tonsil":
                print(os.listdir(f"Intermediary_files/{sample.name}"))
                modified_edgelists = [file for file in os.listdir(f"Intermediary_files/{sample.name}") if "modified" in file and ".csv" in file and "edges" in file]
                print(modified_edgelists)
                for modification in modified_edgelists:
                    modified_sample = analysisSample(sample = sample.name, true_name = f"{sample.name}_{modification[:-4]}", gt_positions=sample.gt_positions_file, specific_edgelist=modification)
                    self.samples.append(modified_sample)
            
        self.load_sample_parameters()
            # modified_sample = analysisSample(sample = sample.name, true_name = f"{sample.name}_modified_nUMI_{fraction}", gt_positions=sample.gt_positions_file, specific_edgelist=f"modified_nUMI_{fraction}.csv")
            # self.samples.append(modified_sample)
            # self.load_sample_parameters()

class analysisSample():
    def __init__(self, sample, gt_positions, specific_edgelist = "all_cells.csv", true_name = None):
        print(sample)
        self.name = sample
        self.edgelist_names = specific_edgelist
        
        if true_name:
            self.name = true_name
        self.gt_positions_file = gt_positions
        self.edges = pd.read_csv(f"Intermediary_files/{sample}/{specific_edgelist}") 
        if os.path.isfile(f"Intermediary_files/{sample}/only_spatial_cells.csv"):
            self.only_spatial_cells_edges = pd.read_csv(f"Intermediary_files/{sample}/only_spatial_cells.csv") 
        else:
            self.only_spatial_cells_edges = None
        
        self.total_umis = self.edges["nUMI"].sum()
        self.load_gt_positions(gt_positions)
        self.calculate_degrees()
        self.n_cells = self.cell_degree_dist[1].sum()
        self.n_beads = self.bead_degree_dist[1].sum()
        self.n_edges = len(self.edges)
        self.n_umis = self.edges["nUMI"].sum()
        if self.only_spatial_cells_edges:
            self.n_cells_gt = self.cell_degree_dist_gt[1].sum()
            self.n_beads_gt = self.bead_degree_dist_gt[1].sum()
            self.n_edges_gt = len(self.only_spatial_cells_edges)
            self.n_umis_gt = self.only_spatial_cells_edges["nUMI"].sum()
        else:
            self.n_cells_gt = None
            self.n_beads_gt = None
            self.n_edges_gt = None
            self.n_umis_gt = None
        if "modified" in specific_edgelist:
            self.modified = true_name
            number = re.findall(r'\d+', specific_edgelist)[0]
            self.fraction = int(number[0])
        else:
            self.modified = False
            # self.load_cell_unipartite()
            

    def load_cell_unipartite(self):
        import networkx as nx

        # Regex pattern
        print("loading unipartite", self.name)
        # Extract match
        # match = re.search(pattern, self.name)
        # if match:
        #     result = match.group(1)  # Get everything after 'N=<number>''
        
        unipartite_path = f"Intermediary_files/{self.name}/{self.edgelist_names[:-4]}_unipartite.csv"
        source_nodes = set(self.edges["cell_bc_10x"].unique())

        if os.path.isfile(unipartite_path):
            print("loading unipartite file")
            self.unipartite_cell = pd.read_csv(unipartite_path)
            print("loaded")
            return
        print("converting to unipartite")

        B = nx.from_pandas_edgelist(self.edges, source="cell_bc_10x", target="bead_bc")
        G = nx.algorithms.bipartite.weighted_projected_graph(B, source_nodes)

        unipartite_df = nx.to_pandas_edgelist(G)
        
        self.unipartite_cell = unipartite_df
        self.unipartite_cell.to_csv(unipartite_path)


    def load_gt_positions(self, gt_positions):
        input_df  = pd.read_csv(f"Input_files/{gt_positions}")
        self.barcodes = input_df["NAME"][1:]
        if self.barcodes.iloc[0][-1] == "1":
            self.barcodes = self.barcodes.str.replace(r'-1$', '', regex=True)
        self.x_coords = input_df["X"][1:].astype("float")
        self.y_coords = input_df["Y"][1:].astype("float")
        self.cell_types = input_df["cell_type"][1:]
        self.gt_df = pd.DataFrame({
                            'x': self.x_coords,
                            'y': self.y_coords,
                            'cell_type': self.cell_types, 
                            "barcode":self.barcodes
                        }).set_index("barcode")

    def calculate_degrees(self):
        edges = self.edges
        self.bead_degrees = np.unique(edges["bead_bc"], return_counts=True)
        self.mean_bead_degrees = np.mean(self.bead_degrees[1])
        self.bead_degree_dist = np.unique(self.bead_degrees[1], return_counts=True)
        self.cell_degrees = np.unique(edges["cell_bc_10x"], return_counts=True)
        self.mean_cell_degrees = np.mean(self.cell_degrees[1])
        self.std_cell_degrees = np.std(self.cell_degrees[1])
        self.cell_degree_dist = np.unique(self.cell_degrees[1], return_counts=True)
        
        edges = self.only_spatial_cells_edges
        if edges:
            self.bead_degrees_gt = np.unique(edges["bead_bc"], return_counts=True)
            self.mean_bead_degrees_gt = np.mean(self.bead_degrees_gt[1])
            self.bead_degree_dist_gt = np.unique(self.bead_degrees_gt[1], return_counts=True)
            self.cell_degrees_gt = np.unique(edges["cell_bc_10x"], return_counts=True)
            self.mean_cell_degrees_gt = np.mean(self.cell_degrees_gt[1])
            self.std_cell_degrees_gt = np.std(self.cell_degrees_gt[1])
            self.cell_degree_dist_gt = np.unique(self.cell_degrees_gt[1], return_counts=True)
        else:
            self.bead_degrees_gt = None
            self.mean_bead_degrees_gt = None
            self.bead_degree_dist_gt = None
            self.cell_degrees_gt = None
            self.mean_cell_degrees_gt = None
            self.std_cell_degrees_gt = None
            self.cell_degree_dist_gt = None


def initialize_samples(all_samples, all_sample_gt_positions):
    all_samples = groupedSamples(all_samples, all_sample_gt_positions)
    return all_samples

def perform_analysis_actions(sample_group:groupedSamples, image_format = "png"):
    sample_group.plot_degree_distributions(format = image_format)
    # sample_group.plot_counts(format= image_format)
    # sample_group.plot_distributions_beads(format= image_format)
    # sample_group.plot_unipartite_edgelength_histogram(format = image_format)
    # sample_group.plot_degrees_per_cell_type(format=image_format)
    plt.show()
if __name__== "__main__":
    total_reads = [1059056528, 1044099392, 943979348]
    total_grepped_reads = [75815144, 8163457, 12501927]
    # total_grepped_reads = [26408956, 12268026, 835345] #unipartite edge
    # all_samples = ["tonsil", "mouse_embryo", "mouse_hippocampus"]
    # all_sample_ground_truth_positions = ["HumanTonsil_spatial.csv", "mouseembryo_spatial.csv", "mousehippocampus_spatial.csv"]
    
    all_samples = ["tonsil"]
    all_sample_ground_truth_positions = ["HumanTonsil_spatial.csv"]

    sample_group = initialize_samples(all_samples, all_sample_ground_truth_positions)
    sample_group.viable_reads = total_grepped_reads
    sample_group.total_reads = total_reads
    for fraction in [125]:
        # sample_group.generate_nUMI_downsampled_samples()
        # sample_group.generate_edges_downsampled_samples(fraction = fraction)
        # sample_group.generate_bead_downsampled_samples()
        pass
    # sample_group.load_modified_samples()
    perform_analysis_actions(sample_group, image_format="pdf")
    