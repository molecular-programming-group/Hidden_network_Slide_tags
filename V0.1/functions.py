import matplotlib.pyplot as plt
import pandas as pd
import os   
import numpy as np
from datetime import datetime
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from quality_metrics import QualityMetrics
import matplotlib.ticker as ticker
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from shapely.affinity import scale
from math import log10, floor
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
import matplotlib
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import seaborn as sns
from scipy.spatial import procrustes

class multi_subgraph_class():
    def __init__(self):
        self.all_subgraphs = []

    def add_subgraph(self, new_concantenated_subgraph):
        self.all_subgraphs.append(new_concantenated_subgraph)

    def get_all_coordinates(self):
        return self.all_subgraphs[0].all_original_coordinates
    
    def get_original_subgraph(self):
        return self.all_subgraphs[0]
    
    def get_final_KNN(self):
        list_of_knns = []
        for subgraph in self.all_subgraphs:
            list_of_knns.append(subgraph.knn[-1])
        return list_of_knns
        pass

    def get_final_CPD(self):
        list_of_cpds = []
        for subgraph in self.all_subgraphs:
            list_of_cpds.append(subgraph.cpd[-1])
        return list_of_cpds
    
    def get_final_dist(self):
        list_of_dists = []
        for subgraph in self.all_subgraphs:
            list_of_dists.append(subgraph.get_average_distortion_all()[-1])
        return list_of_dists
    
class concatenated_subgraph_class():
    def __init__(self):
        self.list_of_alternate_subgraphs =  []
        self.all_original_coordinates = None
        self.knn = []
        self.cpd = []
        self.points_per_area = None

    def add_alternate_subgraph(self, new_subgraph):
        self.list_of_alternate_subgraphs.append(new_subgraph)

    def get_original_subgraph(self):
        return self.list_of_alternate_subgraphs[0]
    
    def add_qm(self, qm_dict):
        self.knn.append(qm_dict["KNN"])
        self.cpd.append(qm_dict["CPD"])

    def get_average_distortion_all(self):
        distortion_list = []
        for subgraph in self.list_of_alternate_subgraphs:
            # dist_normaliser = np.mean(pdist(subgraph.all_true_points))
            distortion_list.append(np.median(subgraph.distortion_all_points))
        return distortion_list
    
    def set_points_density(self):
        subgraph = self.get_original_subgraph

        self.points_per_area = len(subgraph.reconstruction_df.index)/subgraph.hull.area

    def get_max_distortion(self):
        distortion_list = []
        for alt_subgraph in self.list_of_alternate_subgraphs:
            distortion_list.append(alt_subgraph.distortion_all_points)

        return np.max(distortion_list)
    
    def get_second_to_last_subgraph(self):
        if len(self.list_of_alternate_subgraphs)==1:
            return(self.list_of_alternate_subgraphs[0])
        else:
            return self.list_of_alternate_subgraphs[-1]
        
    def make_position_file(self):
        subgraph = self.get_second_to_last_subgraph()
        # print(subgraph.reconstruction_df)
        starting_df = subgraph.reconstruction_df
        starting_df = starting_df.rename(columns = {"x":"raw_recon_x", "y":"raw_recon_y"})
        starting_df["transformed_recon_x"] = subgraph.transformed_reconstruction[:,0]
        starting_df["transformed_recon_y"] = subgraph.transformed_reconstruction[:,1]
        starting_df["true_x"] = subgraph.all_true_points[:,0]
        starting_df["true_y"] = subgraph.all_true_points[:,1]
        starting_df["cell_type"] = subgraph.original_points.loc[subgraph.barcodes.values]["cell_type"]
        print(starting_df)

        starting_df.to_csv(f"{subgraph.location_name}/results/N={len(subgraph.barcodes)}_final_reconstruction_file.csv")

    def make_edge_file(self):
        subgraph = self.get_second_to_last_subgraph()
   
class subgraph_class():
    def __init__(self):
        self.reconstruction_df = None
        self.original_points = None

        self.all_true_points = None
        self.known_recon_points = None
        self.unknown_recon_points = None
        self.colors_points = None
        self.new_recon_points = None
        self.new_true_points = None
        self.outside_points_true = None
        self.outside_points_recon = None
        self.centered_true_points = None

        self.barcodes = None

        self.true_edges = None
        self.recon_edges = None
        self.true_weights = None
        self.recon_weights = None
        self.cell_to_color = None

        self.og_hull = None
        self.dilated_hull = None
        self.dilation_scale = None
        self.qm = None
        self.seq_to_node = None
        self.density = None

        self.filtering_threshold = None
        self.subgraph_counter = None
        self.n_subgraphs = None
        self.current_subplot = None

        self.location_name = None
        self.transformed_reconstruction = None
        self.transformed_edges = None

        self.color_type = None

        self.distortion_all_points = None
        self.collapse_neurons = None
   
    def calculateQM(self):
        self.qm = QualityMetrics(self.all_true_points, self.known_recon_points)
    
    def calculateHull(self):
        self.hull = ConvexHull(self.all_true_points)
        hull_polygon = Polygon([self.hull.points[vertex] for vertex in self.hull.vertices])
        centroid = hull_polygon.centroid  # Centroid of the hull
        dilated_polygon = scale(hull_polygon, xfact=self.dilation_scale, yfact=self.dilation_scale, origin=centroid)
        dilated_vertices = np.array(dilated_polygon.exterior.coords)

        # Create a new convex hull from the dilated vertices
        self.dilated_hull = ConvexHull(dilated_vertices)

    def assessExtendedNodes(self):
        # Each simplex on the hull corresponds to a facet.
        # For each facet we need to check if the test point is on the external side of the facet.
        dilated_hull = self.dilated_hull

        outside_points = []
        outside_points_recon = []

        for point, recon_point in zip(self.all_true_points, self.known_recon_points):
            for simplex in dilated_hull.simplices:
                # Get the vertices for this facet
                vert = dilated_hull.points[simplex]
                
                # Calculate the normal vector to the facet (only in 2D for this example)
                normal = np.array([vert[1][1] - vert[0][1], vert[0][0] - vert[1][0]])
                
                # Choose a point that is known to be inside the hull
                inside_point = np.mean(self.hull.points[self.hull.vertices, :], axis=0)
                # Vector from inside point to one vertex of the facet
                vector_inside_to_vertex = vert[0] - inside_point
                
                # Ensure the normal points outwards by checking its direction relative to the inside point
                if np.dot(normal, vector_inside_to_vertex) < 0:
                    normal = -normal

                # Vector from point on the facet to the test point
                vector_vertex_to_test = point - vert[0]
                # Check if the test point is outside this facet
                if np.dot(normal, vector_vertex_to_test) > 1:
                    outside_points.append(list(point))
                    outside_points_recon.append(list(recon_point))
                    # print(True)
                    break
        self.outside_points_true = np.array(outside_points)
        self.outside_points_recon = np.array(outside_points_recon)

    def findPointsAlignment(self):
        print("Calculating best fitting transformation")
        original_points = self.all_true_points
        transformed_points = self.known_recon_points

        # # Ensure the points are centered by removing the mean
        mean_original = np.mean(original_points, axis=0)
        mean_transformed = np.mean(transformed_points, axis=0)
        centered_original = original_points - mean_original
        centered_transformed = transformed_points - mean_transformed
        self.centered_true_points = centered_original

        # Perform Singular Value Decomposition (SVD) to find the rotation matrix
        U, S, Vt = np.linalg.svd(np.dot(centered_transformed.T, centered_original))

        norm_original = np.linalg.norm(centered_original)
        norm_transformed = np.linalg.norm(centered_transformed)
        scale = norm_original / norm_transformed

        A = np.dot(Vt.T, U.T)*scale

        estimated_original_points = np.dot((transformed_points - mean_transformed), A.T) + mean_original
        estimated_original_points_2 = np.dot((transformed_points), A.T)
        self.transformed_reconstruction = estimated_original_points

        transformed_edges = np.dot(self.recon_edges - mean_transformed, A.T) + mean_original
        self.transformed_edges = transformed_edges            
    
    def getDistortion(self):
        reconstructed_fitted_points = self.transformed_reconstruction
        true_points = self.all_true_points
        distances = [np.linalg.norm(original - reconstructed) for original, reconstructed in
                     zip(true_points, reconstructed_fitted_points)]
        distortion_pairs = [[original, reconstructed] for original, reconstructed in
                     zip(true_points, reconstructed_fitted_points)]
        self.distortion_all_points = distances
        return np.array(distances), np.array(distortion_pairs)

    def getBarcodes(self):
        return self.reconstruction_df.index


def round_to_1(x, sig): # for concentration display
    return round(x, sig-int(floor(log10(abs(x))))-1)

def findFigureByLabel(label):
    """ Check if a figure with a specific label exists and return it if so. """
    figures = [plt.figure(i) for i in plt.get_fignums()]
    for fig in figures:
        if fig.get_label() == label:
            return fig
    return None  # Return None if no figure with the label is found

def DrawKNNandCPD(all_subgraphs, args):
    
    Sample =args["Sample"]
    Sample_run = args["Sample_run"]

    final_knns = all_subgraphs.get_final_KNN()
    final_cpds = all_subgraphs.get_final_CPD()
    final_dists = all_subgraphs.get_final_dist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot for final_knns
    axes[0].bar(range(len(final_knns)), final_knns)
    axes[0].axhline(1, color='red', linestyle='--')
    axes[0].set_ylabel("KNN")
    axes[0].set_ylim([0, 1])
    axes[0].set_title("Final KNNs")
    for i, knn in enumerate(final_knns):
        axes[0].text(i, knn + 0.02, f'{knn:.3f}', ha='center')

    # Plot for final_cpds
    axes[1].bar(range(len(final_cpds)), final_cpds)
    axes[1].axhline(1, color='red', linestyle='--')
    axes[1].set_ylabel("CPD")
    axes[1].set_ylim([0, 1])
    axes[1].set_title("Final CPDs")

    # Plot for final_dists
    axes[2].bar(range(len(final_dists)), final_dists)
    axes[2].axhline(1, color='red', linestyle='--')
    axes[2].set_ylabel("Distortion")
    # axes[2].set_ylim([0, 1])
    axes[2].set_title("Final Distances")
    # Adjust layout
    # plt.tight_layout()
    plt.savefig(f"{Sample}/{Sample_run}/results/KNN_CPD_dist_all_subgraphs.pdf", format = "PDF")
    plt.savefig(f"{Sample}/{Sample_run}/results/KNN_CPD_dist_all_subgraphs.png", format = "png", dpi = 1200)
    for concatenated_subgraph in all_subgraphs.all_subgraphs:
        subgraph = concatenated_subgraph.get_second_to_last_subgraph()
        metrics = subgraph.qm.evaluate_metrics()
        og_dist = metrics["original_distances"]
        recon_dist = metrics["reconstructed_distances"]
        upper_threshold = np.inf  # Change this value as needed

        # Filter based on the upper threshold
        mask = og_dist <= upper_threshold
        og_dist_filtered = og_dist[mask]
        recon_dist_filtered = recon_dist[mask]

        correlation, p_value = pearsonr(og_dist, recon_dist)
        correlation = correlation**2
        mean_observed = np.mean(og_dist)
        ss_tot = np.sum((np.array(og_dist) - mean_observed) ** 2)
        ss_res = np.sum((np.array(og_dist) - np.array(recon_dist)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f'Pearson Correlation: {correlation}, R-squared: {r_squared}, p-value: {p_value}')
        print(f'Pearson Correlation: {correlation}, p-value: {p_value}')

        # Calculate the point density
        xy = np.vstack([og_dist, recon_dist])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        og_dist_sorted = og_dist_filtered[idx]
        recon_dist_sorted = recon_dist_filtered[idx]
        z_sorted = z[idx]

        # Plot scatter plot with density colormap
        fig, ax = plt.subplots(figsize=(10, 10))
        sc = ax.scatter(og_dist, recon_dist, c=z, s=10, cmap='viridis', edgecolor='none', vmin=0)
        plt.colorbar(sc, label='Density')

        # Add correlation line
        m, b = np.polyfit(og_dist, recon_dist, 1)
        plt.plot(og_dist, m * og_dist + b, color='red', linestyle='--')
        ax.set_box_aspect(1)
        # Add labels and title
        plt.xlabel('Original Distances')
        plt.ylabel('Reconstructed Distances')
        plt.title(f'N = {int(len(subgraph.barcodes))}\nPearson Correlation: {correlation:.2f}')
        # plt.savefig(f"{Sample}/{Sample_run}/results/distance_correlation_plot_N={int(len(subgraph.barcodes))}_Pearson Correlation R2={correlation:.2f}.pdf", format ="PDF")
        plt.savefig(f"{Sample}/{Sample_run}/results/distance_correlation_plot_N={int(len(subgraph.barcodes))}_Pearson Correlation R2={correlation:.3f}.png", format ="png", dpi = 1200)

def barDensity(all_subgraphs, arg):

    fig, ax = plt.subplots(1,1)
    all_densities = []
    for concatenated_subgraph in all_subgraphs.all_subgraphs:
        # Your existing code to get KNNs
        density = concatenated_subgraph.get_second_to_last_subgraph().density
        all_densities.append(density)
    print(all_densities)

    ax.bar(range(len(all_densities)), all_densities)
    # ax.axhline(1, color='red', linestyle='--')
    ax.set_ylabel("cell_density, points per µm2")
    # ax.set_ylim([0, 1])
    ax.set_title("Cell densities")

def violinKNN(all_subgraphs, args):
    Sample = args["Sample"]
    Sample_run = args["Sample_run"]
    all_KNNs = []

    # Iterate over concatenated subgraphs
    for concatenated_subgraph in all_subgraphs.all_subgraphs:
        # Your existing code to get KNNs
        KNNs = concatenated_subgraph.get_second_to_last_subgraph().qm.knn_individual
        all_KNNs.append(KNNs)

    # Calculate mean values for each subgraph
    mean_values = [np.mean(KNNs) for KNNs in all_KNNs]

    # Create a bar plot for all subgraphs
    fig, ax = plt.subplots(figsize=(8, 8))
    mean_values = [np.mean(KNNs) for KNNs in all_KNNs]

    sns.boxplot(data=all_KNNs)

    # Mark mean values on the box plot
    for idx, mean_value in enumerate(mean_values):
        plt.plot(idx, mean_value, marker='o', markersize=10, color='red')
        plt.text(idx, mean_value, f'{mean_value:.2f}', ha='center', va='bottom', color='red')

    # Add title and labels
    plt.title('Bar Plot of Mean KNNs for All Subgraphs')
    plt.xlabel('Subgraph')
    plt.ylabel('Mean KNNs')

    ax.set_box_aspect(1)
    ax.set_ylim([0,1.1])
    plt.savefig(f"{Sample}/{Sample_run}/box_knn.pdf", format="pdf")

    fig, ax = plt.subplots(figsize=(8, 8))
    mean_values = [np.mean(KNNs) for KNNs in all_KNNs]

    parts = ax.violinplot(all_KNNs, showmeans=False, showmedians=True)
    for idx, KNNs in enumerate(all_KNNs):
        # Generate random vertical positions
        jitter = np.random.normal(0, 0.05, size=len(KNNs))
        plt.scatter(np.full_like(KNNs, 1) + jitter, KNNs, alpha=0.6, color='k', s= 5, linewidth = 0)

    # # Mark mean values on the box plot
    # for idx, mean_value in enumerate(mean_values):
    #     plt.plot(idx, mean_value, marker='o', markersize=10, color='red', linewidt =0)
    #     plt.text(idx, mean_value, f'{mean_value:.2f}', ha='center', va='bottom', color='red')

    # Add title and labels
    plt.title('Bar Plot of Mean KNNs for All Subgraphs')
    plt.xlabel('Subgraph')
    plt.ylabel('Mean KNNs')

    ax.set_box_aspect(1)
    ax.set_ylim([0,1.1])
        
    # Save or show the plot
    plt.savefig(f"{Sample}/{Sample_run}/violin_knn_N={len(KNNs)}.pdf", format="pdf")
    plt.show()

def plotDensity(all_subgraphs_object):
    print(all_subgraphs_object)
    all_points_df = all_subgraphs_object.get_all_coordinates()

    test = findFigureByLabel("Density highlight")
    
    if test == None:
        fig_density, (ax_density, ax_density_subgraphs) = plt.subplots(1, 2, figsize = (12, 6), label = "Density highlight")
    else:
        fig_density = test
    (ax_density, ax_density_subgraphs) = fig_density.get_axes()
    
    x = all_points_df["X"][1:].astype("float")
    y = all_points_df["Y"][1:].astype("float")
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
    ax_density.scatter(x, y, c = z)
    ax_density_subgraphs.scatter(x, y, c = z)

    for subgraph in all_subgraphs_object.all_subgraphs:
        subgraph_original = subgraph.get_original_subgraph()
        subgraph_x = subgraph_original.all_true_points[:, 0]
        subgraph_y = subgraph_original.all_true_points[:, 1]
        ax_density_subgraphs.scatter(subgraph_x, subgraph_y, s = 25)
        barcodes = subgraph_original.reconstruction_df.index.values
        print(barcodes, "huh")

        print(all_points_df.loc[barcodes])

def dictRandomcellColors(nucleus_coordinates):  
    cell_types = nucleus_coordinates["cell_type"][1:]
    type_to_colour_dict = {}
    for cell_type in cell_types.unique():
        color = list(np.random.choice(range(256), size=3))
        type_to_colour_dict[cell_type] = (color[0]/255,color[1]/255,color[2]/255)
    return type_to_colour_dict

def remapReconstructionToBarcodeFromID(reconstruction, mapping_dict, arguments):
    print(reconstruction)
    if arguments["reconstruction"] =="adaptive":
        idx_to_node = arguments["idx_to_node"].to_dict()["node_ID"]
        reconstruction["node_ID"] = reconstruction["node_ID"].map(idx_to_node)

    reconstruction["node_ID"] = reconstruction["node_ID"].map(mapping_dict)
    barcode_reconstruction = reconstruction.set_index("node_ID")

    return barcode_reconstruction

def generatePlottableEdges(subgraph_edgelist, true_coordinates, reconstructed_coordinates, mapping_dict, arguments):
    all_edges_true = []
    all_edges_re = []
    print(subgraph_edgelist)
    reconstructed_barcodes = reconstructed_coordinates.index.values 
    # print(reconstructed_barcodes)
    if arguments["reconstruction"] =="adaptive":
        idx_to_node = arguments["idx_to_node"].to_dict()["node_ID"]
        subgraph_edgelist["source"] = subgraph_edgelist["source"].map(idx_to_node)
        subgraph_edgelist["target"] = subgraph_edgelist["target"].map(idx_to_node)
    true_edge_weights = []
    recon_edge_weights = []
    for idx, comps in subgraph_edgelist.iterrows():

        cell1 = comps.values[0]
        cell2 = comps.values[1]
        
        if len(comps)==3:
            weight = comps.values[2]

        cell1_bc = mapping_dict[cell1]
        cell2_bc = mapping_dict[cell2]
        check_bc_in_subgraph = cell1_bc in reconstructed_barcodes and cell2_bc in reconstructed_barcodes
        if check_bc_in_subgraph:
            try:
                x_cell1_true = float(true_coordinates.loc[cell1_bc]["X"])
                y_cell1_true = float(true_coordinates.loc[cell1_bc]["Y"])
                x_cell2_true = float(true_coordinates.loc[cell2_bc]["X"])
                y_cell2_true = float(true_coordinates.loc[cell2_bc]["Y"])
                all_edges_true.append([[x_cell1_true, y_cell1_true], [x_cell2_true, y_cell2_true]])
                if len(comps) ==3:
                    true_edge_weights.append(weight)
                
            except:
                pass
            x_cell1_re = float(reconstructed_coordinates.loc[cell1_bc]["x"])
            y_cell1_re = float(reconstructed_coordinates.loc[cell1_bc]["y"])
            x_cell2_re = float(reconstructed_coordinates.loc[cell2_bc]["x"])
            y_cell2_re = float(reconstructed_coordinates.loc[cell2_bc]["y"])
            all_edges_re.append([[x_cell1_re, y_cell1_re], [x_cell2_re, y_cell2_re]])
            if len(comps) ==3:
                recon_edge_weights.append(weight)

    if len(comps) ==3:
        return np.array(all_edges_true), np.array(all_edges_re), true_edge_weights, recon_edge_weights
    else:
        return np.array(all_edges_true), np.array(all_edges_re)

def initializeFigure(rows, n_subgraphs, fig_title, true_coordinates):
    max_figwidth = n_subgraphs*3+2
    if n_subgraphs==1:
        max_figwidth=12
    fig_height = rows*max_figwidth/(n_subgraphs+2)*0.8    

    fig = plt.figure(figsize = (max_figwidth,fig_height), label = fig_title)
    gs = GridSpec(rows, n_subgraphs+2, figure=fig)

    ax_position = fig.add_subplot(gs[:2, :2])
    ax_position.set_xticks([])
    ax_position.set_yticks([])
    ax_position.set_box_aspect(1)
    ax_position.scatter(true_coordinates[1:]["X"].astype("float"), true_coordinates[1:]["Y"].astype("float"), c = "0.8", s = 5)
    return fig, ax_position, gs 

def initializeCurrentSubgraphPlots(fig, gs, subplot_positions_row, which_reconstructions):
    ax_current_subgraph = fig.add_subplot(gs[0, subplot_positions_row])
    ax_current_subgraph.set_xticks([])
    # ax_current_subgraph.set_aspect('equal')
    ax_current_subgraph.set_box_aspect(1)
    ax_current_subgraph.set_yticks([])
    ax_current_reconstruction = fig.add_subplot(gs[2, subplot_positions_row])
    ax_current_reconstruction.set_xticks([])
    ax_current_reconstruction.set_yticks([])
    # ax_current_reconstruction.set_aspect('equal')
    ax_current_reconstruction.set_box_aspect(1)
    ax_linear_transformation = fig.add_subplot(gs[1, subplot_positions_row])
    ax_linear_transformation.set_xticks([])
    ax_linear_transformation.set_yticks([])
    # ax_linear_transformation.set_aspect('equal')
    ax_linear_transformation.set_box_aspect(1)
    return ax_current_subgraph,ax_current_reconstruction, ax_linear_transformation
    
def constructSortedReconstructionPathsDF(original_subgraph_edgefile_path, all_subgraph_reconstructions, argumements):
    dict_for_subgraph_sorting = {}
    print(all_subgraph_reconstructions)

    if argumements["reconstruction"] == "node2vec":
        all_edgelists_filepath = original_subgraph_edgefile_path[:original_subgraph_edgefile_path.find("edge_list")]+"edge_lists/"
        all_reconstructions_filepath = original_subgraph_edgefile_path[:original_subgraph_edgefile_path.find("edge_list")]+"reconstructed_positions/"

        
        #finding correspoding edgefiles for each subgraph
        for subgraph_reconstruction_path in all_subgraph_reconstructions:
            print(subgraph_reconstruction_path) 
            if subgraph_reconstruction_path =="positions_old_index_N=286_dim=2_experimental_edge_list_nbead_7_filtering_STRND.csv":
                subgraph_edgelist = pd.read_csv(all_edgelists_filepath+"old_index_edge_list_N=286_dim=2_experimental_edge_list_nbead_7_filtering.csv")
                subgraph_reconstruction_path = all_reconstructions_filepath+"positions_old_index_N=286_dim=2_experimental_edge_list_nbead_7_filtering_STRND.csv"
                subgraph_edgelist_filename = "old_index_edge_list_N=286_dim=2_experimental_edge_list_nbead_7_filtering.csv"
            else:
                subgraph_edgelist_filename="old_index_edge_list_"+subgraph_reconstruction_path[subgraph_reconstruction_path.find('N='):subgraph_reconstruction_path.find('_STRND')]+".csv"    
                subgraph_edgelist = pd.read_csv(all_edgelists_filepath+subgraph_edgelist_filename)
                print(subgraph_edgelist_filename)

            dict_for_subgraph_sorting.update({subgraph_reconstruction_path:[len(subgraph_edgelist.index), all_edgelists_filepath+subgraph_edgelist_filename]})
    elif argumements["reconstruction"] == "adaptive":
        all_edgelists_filepath = original_subgraph_edgefile_path[:original_subgraph_edgefile_path.find("edge_list")]
        all_reconstructions_filepath = original_subgraph_edgefile_path[:original_subgraph_edgefile_path.find("edges")]+"reconstructions/"
        original_edge_file_name = original_subgraph_edgefile_path.split("/")[-1]
        for subgraph_reconstruction_path in all_subgraph_reconstructions:
            subgraph_edgelist_filename= original_edge_file_name
            subgraph_edgelist = pd.read_csv(all_edgelists_filepath+subgraph_edgelist_filename)
            dict_for_subgraph_sorting.update({subgraph_reconstruction_path:[len(subgraph_edgelist.index), all_edgelists_filepath+subgraph_edgelist_filename]})

    subgraphs_sorted_by_edges = pd.DataFrame.from_dict(dict_for_subgraph_sorting, orient = "index").rename(columns ={"index":"reconstruction", 0:"n_edges", 1:"edges_path"}).sort_values(by="n_edges")    
    return subgraphs_sorted_by_edges

def generateTrueCoordColorPairing(original_nodes, subgraph_variables):
    color_type = subgraph_variables.color_type
    all_true_coordinates = subgraph_variables.original_points
    reconstructed_coordinates = subgraph_variables.reconstruction_df
    mapping_dict = subgraph_variables.seq_to_node
    cell_to_color_dict = subgraph_variables.cell_to_color
    # print(all_true_coordinates)
    cell_in_reconstruction_true = all_true_coordinates.loc[all_true_coordinates.index.isin(reconstructed_coordinates.index)]
    min_y_true  = np.min(cell_in_reconstruction_true["Y"].astype("float"))
    min_x_true = np.min(cell_in_reconstruction_true["X"].astype("float"))
    max_color_true = np.max(cell_in_reconstruction_true["X"].astype("float"))-min_x_true

    true_coordinates = []
    reconstructed_coordinates_known = []
    reconstructed_coordinates_unknown = []

    all_colors = []
    new_node_coordinates_recon = []
    new_node_coordinates_true = []
    if original_nodes is None:
        original_nodes = list(mapping_dict.keys())

    for id, data in reconstructed_coordinates.iterrows():

        truth_available = id in all_true_coordinates.index

        if truth_available:
            y = float(all_true_coordinates.loc[id]["Y"])
            x = float(all_true_coordinates.loc[id]["X"])

            if color_type=="gray":
                color_true = (x-min_x_true)/max_color_true
                color = color_true
                lower_color_limit = 0.3
                color_test = lower_color_limit +(1-lower_color_limit)*(color-lower_color_limit)
                all_colors.append(str(color))

            elif color_type=="cell":
                cell_type = all_true_coordinates.loc[id]["cell_type"]
                # print(subgraph_variables.collapse_neurons)
                if subgraph_variables.collapse_neurons == True:
                    # print(cell_type)
                    if "Neuron" in cell_type:
                        cell_type = "Neuron"
                else: 
                    pass
                color = cell_to_color_dict[cell_type]

                all_colors.append(color)
            elif color_type=="knn":
                pass
            elif color_type == "something else":
                pass
            true_coordinates.append([x, y])
            reconstructed_coordinates_known.append([data["x"], data["y"]])
            
        else:
            reconstructed_coordinates_unknown.append([data["x"], data["y"]])

        if id not in original_nodes:
            new_node_coordinates_recon.append([data["x"], data["y"]])
            new_node_coordinates_true.append([x, y])
    # print(all_colors)
    subgraph_variables.all_true_points = np.array(true_coordinates)
    subgraph_variables.known_recon_points = np.array(reconstructed_coordinates_known)
    subgraph_variables.unknown_recon_points = np.array(reconstructed_coordinates_unknown)
    subgraph_variables.colors_points = all_colors
    subgraph_variables.new_recon_points = np.array(new_node_coordinates_recon)
    subgraph_variables.new_true_points = np.array(new_node_coordinates_true)
    
    return subgraph_variables, original_nodes


    return np.array(outside_points)

def refilteredEdgelistGeneration(edgelist_path, additional_edges_path):
    additional_edgelist = pd.read_csv(additional_edges_path)
    subgraph_edgelist = pd.read_csv(edgelist_path)
    all_subgraph_nodes = subgraph_edgelist.stack().unique()
    extra_edges =additional_edgelist[additional_edgelist["source"].isin(all_subgraph_nodes) & additional_edgelist["target"].isin(all_subgraph_nodes)]
    print(extra_edges.values)
    list_of_edges = list(extra_edges.values)
    print(extra_edges)
    print(subgraph_edgelist)
    print(len(extra_edges) == len(subgraph_edgelist))
    print(edgelist_path)

    if len(extra_edges) == len(subgraph_edgelist)/2 or len(extra_edges) == len(subgraph_edgelist):
        extra_edges.to_csv(edgelist_path, index = None)

        return edgelist_path
    else:
        extra_edges.to_csv(f"{additional_edges_path[:-4]}_refilter_{edgelist_path[edgelist_path.find("N="):]}", index = None)

        return f"{additional_edges_path[:-4]}_{edgelist_path[edgelist_path.find("N="):]}"

def plotReconstruction(fig, gs, ax_current_subgraph, ax_dist, ax_pos, ax_lin, ax_title, subgraph_variables):
    print(f"plotting {subgraph_variables.color_type} colored reconstruction")
    
    colormap = plt.get_cmap('viridis')
    all_true_coordinates = subgraph_variables.all_true_points
    reconstructed_coordinates_array_known = subgraph_variables.known_recon_points
    reconstructed_coordinates_unknown_array = subgraph_variables.unknown_recon_points
    all_colors = subgraph_variables.colors_points
    
    true_edges = subgraph_variables.true_edges
    reconstructed_edges = subgraph_variables.recon_edges
    transformed_edges = subgraph_variables.transformed_edges

    distortions, distortion_pairs = subgraph_variables.getDistortion()
    
    if subgraph_variables.recon_weights !=None:
        # true_w_mean = 20#np.median(subgraph_variables.true_weights)
        # print("calculating edge strength")
        # color_list_true = [1 if w > true_w_mean else w/true_w_mean for w in subgraph_variables.true_weights]
        # # print(color_list_true)
        # color_list_recon = [1 if w > np.median(subgraph_variables.recon_weights) else w/np.median(subgraph_variables.recon_weights) for w in subgraph_variables.recon_weights]
        
        # true_edges = LineCollection(true_edges, color = "k", linewidth = 1, alpha = color_list_true)
        # reconstructed_edges = LineCollection(reconstructed_edges, color = "k", linewidth = 0.5, alpha = color_list_recon)
        # transformed_edges = LineCollection(transformed_edges, color = "k", linewidth = 0.5, alpha = color_list_recon)
        true_edges = LineCollection(true_edges, color = "gray", linewidth = 0.1, alpha = 0.5)
        reconstructed_edges = LineCollection(reconstructed_edges, color = "gray", linewidth = 0.1, alpha = 0.5)
        transformed_edges_for_flipped = LineCollection(transformed_edges, color = "gray", linewidth = 0.1, alpha = 0.5)
    else:
        true_edges = LineCollection(true_edges, color = "black", linewidth = 0.1, alpha = 0.1)
        reconstructed_edges = LineCollection(reconstructed_edges, color = "black", linewidth = 0.1, alpha = 0.1)
        transformed_edges_for_flipped = LineCollection(transformed_edges, color = "black", linewidth = 0.1, alpha = 0.1)
        # transformed_edges_for_dist = LineCollection(transformed_edges, color = "gray", linewidth = 0.1, alpha = 0.5)

    ax_current_subgraph.add_collection(true_edges)
    ax_lin.add_collection(transformed_edges_for_flipped)

    if subgraph_variables.current_subplot ==2:
        ax_pos.scatter(all_true_coordinates[:,0], all_true_coordinates[:,1], c = all_colors, s = 10, edgecolor = "k", linewidths =0.5, zorder =0)

    barcodes = subgraph_variables.barcodes
    subgraph_celltypes = subgraph_variables.original_points["cell_type"][barcodes].values
    if subgraph_variables.collapse_neurons ==True:
        subgraph_celltypes = ['Neuron' if 'Neuron' in cell_type else cell_type for cell_type in subgraph_celltypes]
    if not plt.fignum_exists(f"cellcounts_N={len(subgraph_celltypes)}"):
        plt.figure(f"cellcounts_N={len(subgraph_celltypes)}")
        cell_type_counts = pd.Series(subgraph_celltypes).value_counts()
        ax = cell_type_counts.plot(kind='bar', color='skyblue')

        # Adding counts to the plot
        for i, count in enumerate(cell_type_counts.values):
            ax.text(i, count + 0.1, str(count), ha='center', va='bottom')

        plt.xlabel('Cell Type')
        plt.ylabel('Count')
        plt.title('Count of Each Cell Type')
        plt.xticks(rotation=45, ha='right')
        # plt.tight_layout()
        plt.title(f"N={len(subgraph_celltypes)}")
        plt.savefig(f"{subgraph_variables.location_name}/results/N={len(subgraph_celltypes)}_cellcount_collapse_neurons={subgraph_variables.collapse_neurons}.pdf", format = "pdf")

    estimated_true_points = subgraph_variables.transformed_reconstruction

    ax_lin.scatter(estimated_true_points[:,0], estimated_true_points[:,1], c = all_colors, s = 15, edgecolor = "k", linewidths =0.5, zorder =3)
    ax_current_subgraph.scatter(all_true_coordinates[:,0], all_true_coordinates[:,1], c = all_colors, s = 15, edgecolor = "k", linewidths =0.5, zorder =3)
    sorted_indices = np.argsort(distortions)
    distortion_pairs = distortion_pairs[sorted_indices]
    distortions = distortions[sorted_indices]

    # Normalize distortion values for color mapping
    norm = Normalize(vmin=0, vmax=np.max(distortions))

    # Create LineCollection with specified colormap and normalized distortion values
    distortion_lines = LineCollection(distortion_pairs, linewidth=2, cmap=colormap, norm=norm)
    distortion_lines.set_array(distortions)
    ax_dist.add_collection(distortion_lines)
    # cbar = plt.colorbar(distortion_lines, ax=ax_dist)
    # cbar.set_label('Distortion Value')

    ax_dist.scatter(estimated_true_points[:,0], estimated_true_points[:,1], c = "r")
    ax_dist.scatter(all_true_coordinates[:,0], all_true_coordinates[:,1], c = "b")

    # for i in range(len(estimated_true_points[:,0])):
    #     ax_lin.text(estimated_true_points[:,0][i], estimated_true_points[:,1][i], f'{i}:({estimated_true_points[:,0][i]:.1f}, {estimated_true_points[:,1][i]:.1f})', fontsize=0.5, ha='right')
    # for i in range(len(estimated_true_points[:,0])):
    #     ax_current_subgraph.text(all_true_coordinates[:,0][i], all_true_coordinates[:,1][i], f'{i}:({all_true_coordinates[:,0][i]:.1f}, {all_true_coordinates[:,1][i]:.1f})', fontsize=0.5, ha='right')

    if len(reconstructed_coordinates_unknown_array)!=0:
        ax_dist.scatter(reconstructed_coordinates_unknown_array[:,0], reconstructed_coordinates_unknown_array[:,1], c = "r", s = 3, edgecolor = "k", linewidths =0.5, zorder =1)
    ax_current_subgraph.set_title(ax_title, fontsize = 8)

    # print(subgraph_variables.cell_to_color)
    # quit()
    ax_label = fig.add_subplot(gs[-1, :2])
    ax_label.axis('off')
    legend_elements = [Patch(facecolor=color, edgecolor='k', label=f'{category}')
                   for category, color in subgraph_variables.cell_to_color.items() if category in subgraph_celltypes]
    # ax_label.legend(handles=legend_elements, loc='center', fontsize=10)
    ax_label.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize=10, frameon=False)
    # plt.show()

def plotKNN(ax_current_subgraph, ax_current_reconstruction, ax_pos, ax_lin, ax_title, subgraph_variables):
    print(f"plotting KNN colored reconstruction")
    all_true_coordinates = subgraph_variables.all_true_points
    reconstructed_coordinates_array_known = subgraph_variables.known_recon_points
    reconstructed_coordinates_unknown_array = subgraph_variables.unknown_recon_points
    colormap = plt.get_cmap('viridis')

    true_edges = subgraph_variables.true_edges
    reconstructed_edges = subgraph_variables.recon_edges
    transformed_edges = subgraph_variables.transformed_edges

    true_edges = LineCollection(true_edges, color = "gray", linewidth = 0.1, alpha = 0.5)
    reconstructed_edges = LineCollection(reconstructed_edges, color = "gray", linewidth = 0.1, alpha = 0.5)
    transformed_edges = LineCollection(transformed_edges, color = "gray", linewidth = 0.1, alpha = 0.5)
    ax_current_subgraph.add_collection(true_edges)
    ax_current_reconstruction.add_collection(reconstructed_edges)
    ax_lin.add_collection(transformed_edges)
    
    true_scatter = ax_current_subgraph.scatter(all_true_coordinates[:,0], all_true_coordinates[:,1], vmin=0, vmax=1, 
                                                c = subgraph_variables.qm.knn_individual, cmap = colormap, s = 25, edgecolor = "k", linewidths =0.5, zorder =2)
    ax_current_reconstruction.scatter(reconstructed_coordinates_array_known[:,0], reconstructed_coordinates_array_known[:,1], c = subgraph_variables.qm.knn_individual, cmap = colormap, s = 25, edgecolor = "k", linewidths =0.5, zorder =2)
    # cbar = plt.colorbar(recon_scatter, ax=ax_current_subgraph)
    # cbar.set_label('per points KNN similarity')


    estimated_true_points = subgraph_variables.transformed_reconstruction
    recon_scatter = ax_lin.scatter(estimated_true_points[:,0], estimated_true_points[:,1], vmin=0, vmax=1, 
                                   c = subgraph_variables.qm.knn_individual, cmap = colormap, s = 25, edgecolor = "k", linewidths =0.5, zorder =2)
    # cbar = plt.colorbar(recon_scatter, ax=ax_lin)
    # cbar.set_label('per points KNN similarity')

    if subgraph_variables.current_subplot ==2:
        ax_pos.scatter(all_true_coordinates[:,0], all_true_coordinates[:,1], c = subgraph_variables.qm.knn_individual, s = 10, edgecolor = "k", linewidths =0.5, zorder =3)
    
    if len(reconstructed_coordinates_unknown_array)!=0:
        ax_current_reconstruction.scatter(reconstructed_coordinates_unknown_array[:,0], reconstructed_coordinates_unknown_array[:,1], c = "r", s = 3, edgecolor = "k", linewidths =0.5, zorder =1)
    
    ax_current_subgraph.set_title(ax_title, fontsize = 8)

def plotAllRefilteredSubgraphs(nucleus_coordinates, seq_to_node,original_subgraph_edgefile_path, all_subgraph_reconstructions, arguments):
    which_reconstructions = arguments["which_reconstructions"] 
    color_type = arguments["color_type"]
    cell_to_color_dict = arguments["cell_to_color_dict"]
    dilation_scale = arguments["dilation_scale"]
    plot_knn = arguments["plot_KNN"]
    n_subgraphs = len(all_subgraph_reconstructions)
    original_nodes = None
    original_edges = None
    original_hull = None
    original_dilated_hull = None
    seq_to_node_dict = seq_to_node.to_dict()["cell"]

    rows = 3

    fig_basic, ax_position_basic, gs_basic = initializeFigure(rows, n_subgraphs, color_type+" coloring",nucleus_coordinates)
    subgraph_position_plots = [ax_position_basic]
    if plot_knn == True:
        fig_knn, ax_position_knn, gs_knn = initializeFigure(rows, n_subgraphs, "KNN colored",nucleus_coordinates)
        subgraph_position_plots.append(ax_position_knn)
    
    subgraphs_sorted_by_edges = constructSortedReconstructionPathsDF(original_subgraph_edgefile_path,all_subgraph_reconstructions, arguments)

    subplot_positions_row = 1#n_subgraphs
    # print(subgraphs_sorted_by_edges)
    # for x in subgraphs_sorted_by_edges["edges_path"]:
    #     print(x)
    # quit()
    full_subgraphs = concatenated_subgraph_class()
    full_subgraphs.all_original_coordinates = nucleus_coordinates
    for subgraph_reconstruction_path, (n_edges, edges_path) in subgraphs_sorted_by_edges.iterrows():

        subgraph_variables = subgraph_class()
        subgraph_variables.cell_to_color = cell_to_color_dict
        subgraph_variables.seq_to_node = seq_to_node_dict
        subgraph_variables.dilation_scale = dilation_scale
        subgraph_variables.original_points = nucleus_coordinates
        subgraph_variables.hull = original_hull
        subgraph_variables.dilated_hull = original_dilated_hull
        subgraph_variables.n_subgraphs = n_subgraphs
        subgraph_variables.color_type = color_type
        subgraph_variables.collapse_neurons = arguments["collapse_neurons"]

        edge_path_split = edges_path.split("/")

        subgraph_variables.location_name = f"{edge_path_split[0]}/{edge_path_split[1]}"

        print(subgraph_reconstruction_path)
        subplot_positions_row +=1
        subgraph_variables.current_subplot = subplot_positions_row

        subgraph_edgelist = pd.read_csv(edges_path)
        subgraph_reconstruction = pd.read_csv(subgraph_reconstruction_path)
        bc_reconstruction = remapReconstructionToBarcodeFromID(subgraph_reconstruction,seq_to_node_dict, arguments)

        subgraph_variables.reconstruction_df = bc_reconstruction
        subgraph_variables.barcodes = bc_reconstruction.index
        if subplot_positions_row == 2:
            original_nodes = bc_reconstruction.index

        ax_current_subgraph,ax_current_reconstruction, ax_lin_basic = initializeCurrentSubgraphPlots(fig_basic, gs_basic, subplot_positions_row, which_reconstructions)
        all_axes = [(ax_current_subgraph, ax_current_reconstruction, ax_lin_basic)]
        if plot_knn == True:
            ax_KNN_true, ax_KNN_recons, ax_lin_KNN = initializeCurrentSubgraphPlots(fig_knn, gs_knn, subplot_positions_row, which_reconstructions)
            all_axes.append((ax_KNN_true, ax_KNN_recons, ax_lin_KNN))

        all_edges_true, all_edges_re =  generatePlottableEdges(subgraph_edgelist, nucleus_coordinates, bc_reconstruction, seq_to_node_dict, arguments)

        subgraph_variables.true_edges = all_edges_true
        subgraph_variables.recon_edges = all_edges_re
        
        print(len(subgraph_edgelist), "Edges succesfully drawn")

        # true_coordinates_array, reconstructed_coordinates_array_known,all_colors, reconstructed_coordinates_unknown_array, new_node_coordinates_recon_array, new_node_coordinates_true = generateTrueCoordColorPairing(nucleus_coordinates,bc_reconstruction,seq_to_node_dict, original_nodes, color_type, cell_to_color_dict)
        # print(subgraph_variables.reconstruction_df)
        subgraph_variables, original_nodes = generateTrueCoordColorPairing(original_nodes, subgraph_variables)

        subgraph_variables.calculateQM() 
        
        og_metrics_dict = subgraph_variables.qm.evaluate_metrics()

        full_subgraphs.add_qm(og_metrics_dict)


        if "converted" in edges_path:
            filtering_threshold = edges_path[edges_path.find("_converted")-1]
        else:
            filtering_threshold = edges_path[edges_path.find("nbead_")+6]

        subgraph_variables.filtering_threshold = filtering_threshold
        
        if subplot_positions_row == 2:

            original_filter = filtering_threshold
            subgraph_variables.calculateHull()
            original_hull = subgraph_variables.hull
            original_dilated_hull = subgraph_variables.dilated_hull

            ax_title = f"Original filter: {original_filter}\nKNN: {round_to_1(og_metrics_dict["KNN"], 2)}, CPD: {round_to_1(og_metrics_dict["CPD"], 2)}\nPoints per true area: {round_to_1(len(bc_reconstruction.index)/subgraph_variables.hull.area, 3)}"
            density = len(bc_reconstruction.index)/subgraph_variables.hull.area
            original_N_nodes = len(original_nodes)
            original_edges = n_edges/2

            for ax_pos, (ax_true, ax_recon, ax_lin) in zip(subgraph_position_plots, all_axes):
                # ax_true.set_ylabel("True positions")
                # ax_recon.set_ylabel("Distortion", fontsize = 8)
                # ax_lin.set_ylabel("Transformed\nReconstructions", fontsize = 8)
                for simplex in subgraph_variables.hull.simplices:
                    pass
                    # ax_true.plot(subgraph_variables.all_true_points[simplex, 0], subgraph_variables.all_true_points[simplex, 1], 'k-')
            
        else:
            if which_reconstructions =="extension":
                ax_title = f"Refilter: {filtering_threshold}\nKNN:{round_to_1(og_metrics_dict["KNN"], 2)}, CPD:{round_to_1(og_metrics_dict["CPD"], 2)}\n{len(bc_reconstruction.index)-len(original_nodes)} new nodes"
            else:
                ax_title = f"Refilter: {filtering_threshold}\nKNN:{round_to_1(og_metrics_dict["KNN"], 2)}, CPD:{round_to_1(og_metrics_dict["CPD"], 2)}\n{n_edges-original_edges} new edges"
        subgraph_variables.density = density
        subgraph_variables.findPointsAlignment()
        subgraph_variables.getDistortion()

        if arguments["plot"] ==True:
            plotReconstruction(fig_basic, gs_basic, ax_current_subgraph, ax_current_reconstruction, ax_position_basic, ax_lin_basic, ax_title, subgraph_variables)
            if plot_knn == True:
                plotKNN(ax_KNN_true, ax_KNN_recons, ax_position_knn, ax_lin_KNN, ax_title, subgraph_variables)
        print()
        full_subgraphs.points_per_area = len(bc_reconstruction.index)/subgraph_variables.hull.area
        full_subgraphs.add_alternate_subgraph(subgraph_variables)
    
    if arguments["save_plots"] ==True & arguments["plot"] ==True:
        fig_basic.savefig(f"{subgraph_variables.location_name}/results/subgraph_N={original_N_nodes}_original_filter={original_filter}_summarized_{which_reconstructions}_coloring={color_type}.png", format="PNG", dpi = 1200, transparent=True)
        if plot_knn == True:
            fig_knn.savefig(f"{subgraph_variables.location_name}/results/subgraph_N={original_N_nodes}_original_filter={original_filter}_summarized_{which_reconstructions}_coloring=KNN.png", format="PNG")
    if arguments["save_vector_file"] ==True:
        fig_basic.savefig(f"{subgraph_variables.location_name}/results/subgraph_N={original_N_nodes}_original_filter={original_filter}_summarized_{which_reconstructions}_coloring={color_type}.pdf", format="PDF")
        if plot_knn == True:
            fig_knn.savefig(f"{subgraph_variables.location_name}/results/subgraph_N={original_N_nodes}_original_filter={original_filter}_summarized_{which_reconstructions}_coloring=KNN.pdf", format="PDF")
    return full_subgraphs
    