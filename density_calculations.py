import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from datetime import datetime
import re
from timeit import default_timer as timer
from matplotlib.collections import LineCollection
from scipy.stats import gaussian_kde
import networkx as nx
import community as community_louvain
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

import matplotlib as mpl


class plotting_object():
    def __init__(self, sample, run):
        
        self.sample = sample
        self.run = run
        self.seq_to_node = None


        self.cell_density_gaussian = None
        self.bead_density_gaussian = None
        self.umi_density_gaussian = None

        self.unipartite_edges = None
        self.edges_df = None
        self.positions_df = None

        self.fig = None
        self.subgraph_files = None

def plotCellDensity(summary_object):

    coordinates_df = summary_object.positions_df
    fig_density, (ax_density, ax_violin) = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots
    print()

    x = coordinates_df["X"][1:].astype("float")
    y = coordinates_df["Y"][1:].astype("float")
    xy = np.vstack([x, y])
    cell_density = gaussian_kde(xy)
    summary_object.cell_density_gaussian = cell_density
    z = cell_density(xy)
    summary_object.xy = xy
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
    
    # Scatter plot with cell densities
    scatter = ax_density.scatter(x, y, c=z)
    ax_density.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax_density)
    cbar.set_label('Cell density')

    # Violin plot for cell densities
    ax_violin.violinplot(z, vert=False)
    ax_violin.set_xlabel('Density')
    ax_violin.set_title('Cell Density Distribution')

    plt.tight_layout()
    # plt.show()
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'cell_density': z,
    })
    summary_object.cell_density = df
    return summary_object

def plotBeadDensity(summary_object): # unused but would check the degree of each cell
    fig_density, ax_density = plt.subplots(1,1, label = "Bead density highlight")
    edges_df = summary_object.edges_df
    count_matrix = edges_df["cell_bc_10x"].value_counts()
    print(len(count_matrix))

    all_points = []
    cells_not_found = 0
    for cell, values in nucleus_coordinates_df[1:].iterrows():
        cell = cell[:-2]
        try:        
            # print(count_matrix[cell])
            all_points.append([cell+"-1", float(values["X"]), float(values["Y"]), count_matrix[cell]])
        except:
            # print("no edges found for cell", cell)
            cells_not_found += 1

    df = pd.DataFrame(all_points, columns= ["bc", "x", "y", "z"])
    df = df.sort_values(by='z', ascending=True)

    print(cells_not_found)
    weights=df["z"]
    x = df["x"]
    y = df["y"]
    # ax_density.scatter(x, y, c = weights)
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, weights=weights)

    # Step 3: Evaluate the KDE at each point
    density_kde = gaussian_kde(xy)

    # KDE for weighted density
    weighted_kde = gaussian_kde(xy, weights=weights)

    # Step 3: Evaluate both KDEs at each point
    density = density_kde(xy)
    weighted_density = weighted_kde(xy)

    df['weighted_density'] = weighted_density

    # Sort by normalized_density to ensure points with highest density are plotted last
    df = df.sort_values(by='weighted_density')

    # Plot the data, ensuring points with higher density are on top
    scatter = ax_density.scatter(df["x"], df["y"], c=df["weighted_density"], cmap='viridis')
    ax_density.set_aspect('equal')
    ax_density.set_aspect('equal')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Degree density')
    summary_object.bead_density_gaussian = weighted_kde
    fig_density.savefig("Mouse_Embryo/20240516_spatial_bead_sum1-256_edge_1_n_connections_1-256/Bead_densities.pdf", format ="PDF")
    print(summary_object.cell_density)
    print(df)
    summary_object.bead_density = df
    return summary_object
    pass

def plotUMIDensity(summary_object):
    fig_density, ax_density = plt.subplots(1,1, label = "UMI density highlight")
    edges_df = summary_object.edges_df
    filtering_df =edges_df.groupby("cell_bc_10x").sum()
    print(filtering_df)

    all_points = []
    cells_not_found = 0
    for cell, values in nucleus_coordinates_df[1:].iterrows():
        cell = cell[:-2]

        try:    
            nUMI = filtering_df.loc[cell]["nUMI"]
            all_points.append([cell+"-1", float(values["X"]), float(values["Y"]), nUMI])
            
        except:
            # print("no edges found for cell", cell)
            cells_not_found += 1
    print(cells_not_found)
    print(len(all_points))

    df = pd.DataFrame(all_points, columns= ["bc", "x", "y", "z"])

    df = df.sort_values(by='z', ascending=True)

    weights=df["z"]
    x = df["x"]
    y = df["y"]
    # ax_density.scatter(x, y, c = weights)
    xy = np.vstack([x, y])

    # Step 3: Evaluate the KDE at each point
    density_kde = gaussian_kde(xy)

    # KDE for weighted density
    weighted_kde = gaussian_kde(xy, weights=weights)

    # Step 3: Evaluate both KDEs at each point
    density = density_kde(xy)
    weighted_density = weighted_kde(xy)

    df['weighted_density'] = weighted_density

    # Sort by normalized_density to ensure points with highest density are plotted last
    df = df.sort_values(by='weighted_density')

    # Plot the data, ensuring points with higher density are on top
    scatter = ax_density.scatter(df["x"], df["y"], c=df["weighted_density"], cmap='viridis')
    ax_density.set_aspect('equal')
    cbar = plt.colorbar(scatter)
    cbar.set_label('UMI density')
    summary_object.umi_density_gaussian = weighted_kde
    summary_object.umi_density = df
    fig_density.savefig("Mouse_Embryo/20240516_spatial_bead_sum1-256_edge_1_n_connections_1-256/UMI_densities.pdf", format ="PDF")
    return summary_object

def normalizedDensities(summary_object): 
    coordinates_df = summary_object.positions_df

    x = coordinates_df["X"][1:].astype("float")
    y = coordinates_df["Y"][1:].astype("float")
    
    xy = np.vstack([x,y])
    cell_density = summary_object.cell_density_gaussian(xy)
    bead_density = summary_object.bead_density_gaussian(xy)
    umi_density = summary_object.umi_density_gaussian(xy)
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'cell_density': cell_density,
        'bead_density': bead_density,
        'umi_density': umi_density
    })
    summary_object.all_densities = df
    return summary_object
    # Calculate density ratios
    df['bead_over_cell_density'] = df['bead_density'] / df['cell_density']
    df['umi_over_cell_density'] = df['umi_density'] / df['cell_density']
    df['umi_over_bead_density'] = df['umi_density'] / df['bead_density']

    # Sort DataFrame by density ratios to ensure highest density points are plotted last
    df_sorted_boc = df.sort_values(by='bead_over_cell_density')
    df_sorted_uoc = df.sort_values(by='umi_over_cell_density')
    df_sorted_uob = df.sort_values(by='umi_over_bead_density')

    fig, (ax_b, ax_u, ax_bu) = plt.subplots(1, 3, figsize = (18,5))
    # Step 4: Normalize the weighted KDE by the point density KDE
    scatter_b = ax_b.scatter(df_sorted_boc['x'], df_sorted_boc['y'], c=df_sorted_boc['bead_over_cell_density'], cmap='viridis')
    ax_b.set_aspect('equal')
    ax_b.set_title("Degree density over cell density")
    cbar_b = plt.colorbar(scatter_b, ax=ax_b)
    cbar_b.set_label('Degree over Cell density')

    # Plot UMI density over cell density
    scatter_u = ax_u.scatter(df_sorted_uoc['x'], df_sorted_uoc['y'], c=df_sorted_uoc['umi_over_cell_density'], cmap='viridis')
    ax_u.set_aspect('equal')
    ax_u.set_title("UMI density over cell density")
    cbar_u = plt.colorbar(scatter_u, ax=ax_u)
    cbar_u.set_label('UMIs over Cell density')

    # Plot UMI density over Degree density
    scatter_bu = ax_bu.scatter(df_sorted_uob['x'], df_sorted_uob['y'], c=df_sorted_uob['umi_over_bead_density'], cmap='viridis')
    ax_bu.set_aspect('equal')
    ax_bu.set_title("UMI density over Degree density")
    cbar_bu = plt.colorbar(scatter_bu, ax=ax_bu)
    cbar_bu.set_label('UMI over Degree density')

    summary_object.fig = fig
    return summary_object

def readSubgraphs(summary):
    all_subgraphs = os.listdir(f"{summary.sample}/{summary.run}/analysis_edges")
    print(all_subgraphs)
    seq_to_node = summary.seq_to_node.to_dict()["cell"]
    all_subgraph_barcodes = []

    for subgraph in all_subgraphs:
        edgelist = pd.read_csv(f"{summary.sample}/{summary.run}/analysis_edges/{subgraph}")
        # print(edgelist)
        unique_values = pd.concat([edgelist['source'], edgelist['target']]).unique()
        subgraph_barcodes = [seq_to_node[node] for node in unique_values]
        print(len(subgraph_barcodes), "huh")
        # print(len(subgraph_barcodes))
        print(subgraph_barcodes)
        all_subgraph_barcodes.extend(subgraph_barcodes)

    return all_subgraph_barcodes

def AnalyseSubplotsDensity(summary_object):
    
    coordinates_df = summary_object.positions_df
    print()
    print()

    fig_density, ((ax_density, ax_violin),(ax_density_bead, ax_violin_bead),(ax_density_umi, ax_violin_umi)) = plt.subplots(3, 2, figsize=(12, 10))  # Create a figure with two subplots
    density_df = summary_object.all_densities
    density_df["bead_over_cell"] = density_df["bead_density"].values/density_df["cell_density"]
    density_df["umi_over_cell"] = density_df["umi_density"].values/density_df["cell_density"]

    df_sorted_cell = density_df.sort_values(by='cell_density')

    df_sorted_boc = density_df.sort_values(by='bead_over_cell')

    df_sorted_uoc = density_df.sort_values(by='umi_over_cell')
    print(density_df)
    scatter = ax_density.scatter(df_sorted_cell["x"], df_sorted_cell["y"], c=df_sorted_cell["cell_density"])
    ax_density.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax_density)
    cbar.set_label('Cell density')

    print(density_df["bead_density"].values/density_df["cell_density"].values)

    scatter = ax_density_bead.scatter(df_sorted_boc["x"], df_sorted_boc["y"], c=df_sorted_boc["bead_over_cell"].values)
    ax_density_bead.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax_density_bead)
    cbar.set_label('Degree density')

    scatter = ax_density_umi.scatter(df_sorted_uoc["x"], df_sorted_uoc["y"], c=df_sorted_uoc["umi_over_cell"].values)
    ax_density_umi.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax_density_umi)
    cbar.set_label('UMI density')

    # Violin plot for cell densities
    # ax_violin.violinplot(density_df["cell_density"].values, vert=False)
    ax_violin.set_xlabel('Density')
    ax_violin.axvline(np.mean(density_df["cell_density"]))
    ax_violin.set_title('Cell Density Distribution')

    # ax_violin_bead.violinplot(density_df["bead_over_cell"], vert=False)
    ax_violin_bead.set_xlabel('Density')
    ax_violin_bead.set_title('Bead Density Distribution')

    # ax_violin_umi.violinplot(density_df["umi_over_cell"], vert=False)
    plt.axvline(np.mean(density_df["umi_over_cell"]))
    ax_violin_umi.set_xlabel('Density')
    ax_violin_umi.set_title('UMI Density Distribution')

    combined_cell_z = []
    combined_bead_z = []
    combined_umi_z = []

    plt.tight_layout()
    combined_cell_z = density_df.loc[summary_object.subgraph_barcodes]["cell_density"].values
    combined_bead_z = density_df.loc[summary_object.subgraph_barcodes]["bead_over_cell"].values
    combined_umi_z = density_df.loc[summary_object.subgraph_barcodes]["umi_over_cell"].values
    
    def rank_biserial_correlation(U, n1, n2):
        return 1 - (2 * U) / (n1 * n2)
    
    mask = density_df.index.isin(summary_object.subgraph_barcodes)
    inverse_mask = ~mask
    filtered_df = density_df[inverse_mask]

    non_sub_cell_z = filtered_df["cell_density"].values
    non_sub_bead_z = filtered_df["bead_over_cell"].values
    non_sub_umi_z = filtered_df["umi_over_cell"].values
    t_stat, p_value = ttest_ind(combined_cell_z, non_sub_cell_z, alternative='greater')
    print(p_value)
    print(len(combined_cell_z), len(non_sub_cell_z))
    t_stat, p_value = ttest_ind(combined_umi_z, non_sub_umi_z, alternative='greater')
    print(p_value)
    print(len(combined_cell_z), len(non_sub_cell_z))
    n1, n2 = len(combined_umi_z), len(non_sub_umi_z)

    u_stat, p_value = mannwhitneyu(combined_cell_z, non_sub_cell_z, alternative='greater')
    print(f"\n\np-value for combined_cell_z vs. non_sub_cell_z: {p_value}")
    print(f"U statistic: {u_stat}")
    print(f"Sample sizes: {len(combined_cell_z)}, {len(non_sub_cell_z)}")
    rank_biserial_corr = rank_biserial_correlation(u_stat, n1, n2)
    print(f"Rank-biserial correlation: {rank_biserial_corr} or {(2 * u_stat) / (n1 * n2) - 1}\n")

    # Perform Mann-Whitney U test for combined_umi_z and non_sub_umi_z
    u_stat, p_value = mannwhitneyu(combined_umi_z, non_sub_umi_z, alternative='greater')
    print(f"p-value for combined_umi_z vs. non_sub_umi_z: {p_value}")
    print(f"U statistic: {u_stat}")
    print(f"Sample sizes: {len(combined_umi_z)}, {len(non_sub_umi_z)}")
    rank_biserial_corr = rank_biserial_correlation(u_stat, n1, n2)
    print(f"Rank-biserial correlation: {rank_biserial_corr} or {(2 * u_stat) / (n1 * n2) - 1}\n")

    from scipy.stats import ks_2samp, levene, brunnermunzel
    
    bm_stat, bm_p_value = brunnermunzel(combined_umi_z, non_sub_umi_z)
    print(f'Brunner-Munzel test statistic: {bm_stat}, p-value: {bm_p_value}')

    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = ks_2samp(combined_umi_z, non_sub_umi_z)
    print(f'Kolmogorov-Smirnov test statistic: {ks_stat}, p-value: {ks_p_value}')

    # Levene's Test for Equality of Variances
    levene_stat, levene_p_value = levene(combined_umi_z, non_sub_umi_z)
    print(f'Levene test statistic: {levene_stat}, p-value: {levene_p_value}')

    num_points_first = len(density_df["cell_density"])
    
    num_points_second = len(summary_object.subgraph_barcodes)
    print(num_points_first, num_points_second)

    # scale_factor = 1
    # Plot the second violin plot with adjusted scale
    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    # Add jittered points function
    def add_jittered_points(ax, data, position, color):
        jitter = np.random.normal(0, 0.04, size=len(data))  # Create random horizontal jitter
        ax.scatter(np.ones_like(data) * position + jitter, data, alpha=0.6, color="k", edgecolor='k', s=5, linewidth = 0)

    # Plot for cell_z variables
    positions_cell = [1, 2]
    axs[0].violinplot(combined_cell_z, positions=[positions_cell[0]], showmeans=False, showmedians=True)
    axs[0].violinplot(non_sub_cell_z, positions=[positions_cell[1]], showmeans=False, showmedians=True)

    # Add vertical lines for the means for cell_z variables
    # axs[0].axvline(positions_cell[0], color='k', linestyle='--', ymax=0.75)
    # axs[0].axvline(positions_cell[1], color='k', linestyle='--', ymax=0.75)

    # Adding jittered points for cell_z variables
    add_jittered_points(axs[0], combined_cell_z, positions_cell[0], 'b')
    add_jittered_points(axs[0], non_sub_cell_z, positions_cell[1], 'b')

    # Customize the x-axis for cell_z subplot
    axs[0].set_xticks(positions_cell)
    axs[0].set_xticklabels(['Combined Cell Z', 'Non-sub Cell Z'])
    axs[0].set_title('Cell Z Variables')

    # Plot for umi_z variables
    positions_umi = [1, 2]
    axs[1].violinplot(combined_umi_z, positions=[positions_umi[0]], showmeans=False, showmedians=True)
    axs[1].violinplot(non_sub_umi_z, positions=[positions_umi[1]], showmeans=False, showmedians=True)

    # Add vertical lines for the means for umi_z variables
    # axs[1].axvline(positions_umi[0], color='b', linestyle='--', ymax=0.75)
    # axs[1].axvline(positions_umi[1], color='b', linestyle='--', ymax=0.75)

    # Adding jittered points for umi_z variables
    add_jittered_points(axs[1], combined_umi_z, positions_umi[0], 'b')
    add_jittered_points(axs[1], non_sub_umi_z, positions_umi[1], 'b')

    # Customize the x-axis for umi_z subplot
    axs[1].set_xticks(positions_umi)
    axs[1].set_xticklabels(['Combined UMI Z', 'Non-sub UMI Z'])
    axs[1].set_title('UMI Z Variables')
    axs[1].set_box_aspect(1)
    axs[0].set_box_aspect(1)
    # Adjust layout for better visualization

    plt.savefig(f"Mouse_Embryo/20240516_spatial_bead_sum1-256_edge_1_n_connections_1-256/violins_and_heatmaps.pdf", format="pdf")
    # plt.show()
Sample = "Mouse_Embryo"
Sample_run = "20240516_spatial_bead_sum1-256_edge_1_n_connections_1-256"

entries = os.listdir(Sample+"/") 
str_id = "spatial"
spatial_file = [entries for entries in entries if str_id in entries][-1]
nucleus_coordinates_df = pd.read_csv(Sample+"/"+spatial_file).set_index("NAME")

filename = "edge_list_filtered_by_per_edge_weight_1.csv"
# filename = "edge_list_nbead_1_f   iltering.csv"
edge_file = f'Mouse_Embryo/20240516_spatial_bead_sum1-256_edge_1_n_connections_1-256/{filename}'  # Ensure this path points to your CSV file


# edge_file = "SRR11_edge_list_sequences_only_spatial.csv" #  SRR11_edge_list_sequences_only_spatial.csv , SRR11_edge_list_sequences_all_CR_barcodes.csv
edges_df = pd.read_csv(edge_file)
unipartite_edges = pd.read_csv(f'Mouse_Embryo/20240516_spatial_bead_sum1-256_edge_1_n_connections_1-256/all_edges_with_weights_and_distances_with_check.csv').drop(["one_is_one"], axis = 1)

summary_object = plotting_object(Sample, Sample_run)
summary_object.seq_to_node = pd.read_csv(f"{Sample}/{Sample}_cell_and_bead-idx_mapping.csv", names=["cell", "node_ID"]).set_index("node_ID")
summary_object.edges_df = edges_df
summary_object.unipartite_edges = unipartite_edges
summary_object.positions_df = nucleus_coordinates_df

entries = os.listdir(f"{Sample}/{Sample_run}/") 
str_id = "final_reconstruction_file.csv"
subgraph_files = [f"{Sample}/{Sample_run}/"+entries for entries in entries if str_id in entries]
summary_object.subgraph_files = subgraph_files

summary_object = plotCellDensity(summary_object)
summary_object = plotBeadDensity(summary_object)
summary_object = plotUMIDensity(summary_object)

summary_object = normalizedDensities(summary_object)
plt.close("all")
subgraph_barcodes = readSubgraphs(summary_object)

summary_object.subgraph_barcodes = subgraph_barcodes
AnalyseSubplotsDensity(summary_object)

# networkCommunityDetectionBMM(summary_object)
# plt.show()