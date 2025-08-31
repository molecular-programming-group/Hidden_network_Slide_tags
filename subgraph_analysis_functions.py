import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re
import numpy as np
import networkx as nx
from matplotlib import cm
from matplotlib.colors import Normalize

from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from Utils import *
from pathlib import Path
from math import log10, floor, ceil
from typing import List
from alphamorph.apply import alphamorph_apply
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm



class subgraphCollection():
    def __init__(self, edgelist_location, config):
        self.subgraphs: List[subgraphToAnalyse] = []
        self.edgelist_location = edgelist_location
        self.config = config

    def add_subgraph(self, subgraph):
        pass

    def get_subgraph_numbers(self):
        subgraph_numbers = []
        for subgraph in self.subgraphs:
            subgraph_numbers.append(subgraph.number)
        self.subgraph_numbers = subgraph_numbers
        return subgraph_numbers

class subgraphToAnalyse(): 
    '''
    This class us used as the hub to perform analysis actions, and it has a great many properties and methods. Essentially run will result in one or more reconstructed subgraph, and this class gathers all its properties and results into one object
    '''
    def __init__(self, config,  filename = None, edgelist = None):
        self.config = config
        self.name = filename
        self.is_modified_subgraph = False
        self.re_filter = False
        if not self.config.plot_modification:
            self.config.modification_type = "unmodified"
        self.filtering_threshold = config.subgraph_to_analyse.threshold
        self.color_set = self.config.vizualisation_args.colours
        
        # self.color_scheme = 
        self.number = re.search(r"subgraph_(\d+)", self.name).group(1)
        self.n_nodes = re.search(r"N=(\d+)", self.name).group(1)
        self.original_edgelist = edgelist
        self.point_ids = self.original_edgelist.stack().unique()
        self.n_all_nodes = len(self.point_ids)
        
        self.edgelist_location = None
        self.base_edgelist = None
        self.all_ground_truth_points = self.config.gt_points.gt_df
        self.colors = self.load_colors()
        self.bc_to_idx = dict(zip(config.idx_to_bc['index'], config.idx_to_bc['barcode'])) # This is for converting numerical identities of ndoes to barcodes 
        self.all_aligned_reconstructions = None

        self.unenriched_flag = not self.config.plot_modification # If the config option is false, this will be true, this signals to not plot or analyse any modifed versions of the base subgraph. If no modifications are available, it will also flip to True in the code
        self.all_enrichedSubgraphs = []
        self.reconstruction_dimension = config.filter_analysis_args.reconstruction_dimension
        self.dstrn_all_reconstructions = None
        self.all_unaligned_morphed_reconstructions = None
        self.create_gt_df()
        self.n_cells = 0
        self.all_reconstructions = None
        self.load_predicted_cell_types()   
        self.pseudo_cells_mapping = None

    def load_predicted_cell_types(self):
        if not self.config.predicted_cell_types_file:
            self.predicted_cell_types = pd.DataFrame(columns = ["node_bc", "prediction_score", "node_type"])
            return
        predicted_cell_types_base = pd.read_csv(f"Input_files/{self.config.predicted_cell_types_file}")

        predicted_cell_types = predicted_cell_types_base.filter(like="node", axis = 1).copy()
        predicted_cell_types.loc[:,"node_bc"] = predicted_cell_types["node_bc"].str[:-2]
        predicted_cell_types.loc[:,"prediction_score"] = predicted_cell_types_base["prediction.score.max"].values

        self.predicted_cell_types = predicted_cell_types.set_index("node_bc")
        

    def genenerate_nx_graph(self, edgelist = None):
        '''
        Some analyses use a networkx version of the network, which is created here 
        '''

        if edgelist ==None:
            edgelist = self.original_edgelist
        self.networkx_graph = nx.Graph()
        print("Creating networkx graph")
        for _, row in edgelist.iterrows():
            self.networkx_graph.add_edge(row['source'], row['target'])
            
    def load_colors(self):
        '''
        This functions loads the cell colors from a python file. If the config-specified set of colours is not found, it will instead generate random colors for each cell type
        '''
        import importlib
        module = importlib.import_module("cell_colors")
        colors = getattr(module, self.color_set)
        if colors == {}:
            colors = dictRandomcellColors(self.all_ground_truth_points)
        return colors
    
    def create_gt_df(self):
        '''
        This method initializes the Slide-tags determined positions for the nodes in the subgraph specifically, which might not always be all nodes
        '''

        print("Creating ground truth positions dataframe")
        bc_to_idx_series = pd.Series(self.bc_to_idx).reindex(self.point_ids)
        valid_bc_to_idx = bc_to_idx_series[bc_to_idx_series.isin(self.config.gt_points.gt_df.index)] # This extracts nodes that actually have a ground truth available

        # Extract ground truth positions
        gt_positions = self.config.gt_points.gt_df.loc[valid_bc_to_idx.values, ["x", "y"]]

        # Create the DataFrame
        self.gt_positions = pd.DataFrame({
            "node_ID": valid_bc_to_idx.index,  # Original node IDs
            "x": gt_positions["x"].values,
            "y": gt_positions["y"].values
        }).set_index("node_ID")

        self.n_gt_cells = len(self.gt_positions)
        self.n_points_with_gt = len(self.gt_positions)

    def find_reconstructions(self):
        '''
        This function uses the edgelist location to find and read all reconstructions corresponding to the specific subgraph
        '''
        # print("Finding reconstructions")
        self.reconstruction_location = replace_first_folder(self.edgelist_location, "Subgraph_reconstructions")+f"_{self.reconstruction_dimension}D"

        self.all_reconstructions = [pd.read_csv(f"{self.reconstruction_location}/{recon}").set_index("node_ID") for recon in os.listdir(self.reconstruction_location) if ".csv" in recon and f"subgraph_{self.number}_" in recon]
        if self.all_reconstructions == []:
            self.point_ids = []
        else:
            self.point_ids = self.all_reconstructions[0].index.values

    def find_enriched_subgraph(self):
        '''
        Since this code is also used to analyse the modified subgraphs, it has to find the same files for the base subgraph for the modified subgraphs.
        Additional types of subgraph modifications would need to be added here.
        Its purpose is to find and read each subgraphs reconstructions as pandas dataframes, which requires some path and folder creative reading
        '''

        if not self.unenriched_flag:
            self.top_n_reconstructions = 1

            if self.config.modification_type =="enriched":
                print("Finding enriched subgraphs")
                self.enriched_edgelists_location = f"{self.edgelist_location}/{self.name[:-4]}_enriched"
                self.enriched_reconstruction_location = replace_first_folder(self.enriched_edgelists_location, "Subgraph_reconstructions")
                if Path(self.enriched_edgelists_location).exists():
                    self.all_enrichedSubgraphs = [enrichedSubgraph(self, f"{self.enriched_edgelists_location}/{enrichment}") for enrichment in os.listdir(self.enriched_edgelists_location)]
                    self.all_enrichedSubgraphs = sorted(self.all_enrichedSubgraphs, key=lambda subgraph: subgraph.enrichment_threshold, reverse=True)
                else:
                    self.unenriched_flag = True

            elif self.config.modification_type == "gated":
                print("Finding gated subgraphs")
                base_edgelist_location = replace_first_folder(self.edgelist_location, "Output_files")+f"_{self.reconstruction_dimension}D"
                if self.config.subgraph_to_analyse.gating_threshold =="all":
                    all_gated_subgraphs = [f for f in os.listdir(base_edgelist_location) if f"_gated_" in f and ".csv" not in f]                
                else:
                    all_gated_subgraphs = [f for f in os.listdir(base_edgelist_location) if f"_gated_{self.config.subgraph_to_analyse.gating_threshold}" in f and ".csv" not in f]

    
                if not self.config.subgraph_to_analyse.include_recursively_gated:
                    all_gated_subgraphs = [file for file in all_gated_subgraphs if len(re.findall("_gated_",file))<2]
                if all_gated_subgraphs:
                    self.all_enrichedSubgraphs = []
                    for gated_subgraph in all_gated_subgraphs:
                        self.enriched_edgelists_location = f"{base_edgelist_location}/{gated_subgraph}"
                        self.enriched_reconstruction_location = replace_first_folder(self.enriched_edgelists_location, "Subgraph_reconstructions")
                        if Path(self.enriched_edgelists_location).exists():
                            enrichment = os.listdir(self.enriched_edgelists_location)[0] #There should always be just one edgelist per gated edgelist folder
                            gated_subgraph = enrichedSubgraph(self, f"{self.enriched_edgelists_location}/{enrichment}")
                            print(f"{self.enriched_edgelists_location}/{enrichment}")
                            n_reconstructions = len(gated_subgraph.all_reconstructions)
                            if n_reconstructions > self.max_reconstructions:
                                self.max_reconstructions = n_reconstructions
                            self.all_enrichedSubgraphs.append(gated_subgraph)
                        
                    self.all_enrichedSubgraphs = sorted(self.all_enrichedSubgraphs, key=lambda subgraph: subgraph.enrichment_threshold, reverse=True)
                else:
                    self.unenriched_flag = True

            elif self.config.modification_type == "dbscan":
                print("Finding dbscan gated subgraphs")
                base_edgelist_location = replace_first_folder(self.edgelist_location, "Output_files")+f"_{self.reconstruction_dimension}D"
                if self.config.subgraph_to_analyse.gating_threshold =="dbscan" or self.config.subgraph_to_analyse.gating_threshold =="all":
                    all_gated_subgraphs = [f for f in os.listdir(base_edgelist_location) if f"_dbscan_" in f and ".csv" not in f]                
                else:
                    all_gated_subgraphs = [f for f in os.listdir(base_edgelist_location) if f"{self.config.subgraph_to_analyse.gating_threshold}" in f and ".csv" not in f]
                if not self.config.subgraph_to_analyse.include_recursively_gated:
                    all_gated_subgraphs = [file for file in all_gated_subgraphs if len(re.findall("_dbscan_",file))<2]
                if all_gated_subgraphs:
                    self.all_enrichedSubgraphs = []
                    for gated_subgraph in all_gated_subgraphs:
                        self.enriched_edgelists_location = f"{base_edgelist_location}/{gated_subgraph}"
                        self.enriched_reconstruction_location = replace_first_folder(self.enriched_edgelists_location, "Subgraph_reconstructions")
                        if Path(self.enriched_edgelists_location).exists():
                            enrichment = os.listdir(self.enriched_edgelists_location)[0] #There should always be just one edgelist per gated edgelist folder
                            print(f"{self.enriched_edgelists_location}/{enrichment}")
                            gated_subgraph = enrichedSubgraph(self, f"{self.enriched_edgelists_location}/{enrichment}")
                            n_reconstructions = len(gated_subgraph.all_reconstructions)
                            if n_reconstructions > self.max_reconstructions:
                                self.max_reconstructions = n_reconstructions
                            self.all_enrichedSubgraphs.append(gated_subgraph)
                        
                    self.all_enrichedSubgraphs = sorted(self.all_enrichedSubgraphs, key=lambda subgraph: subgraph.enrichment_threshold, reverse=True)
                else:
                    self.unenriched_flag = True

    def align_reconstructions_to_gt_svd(self):
        '''
        This methods takes the reconstructed points, and aligns them to the ground truth positions . 
        '''

        all_aligned_reconstructions = []
        gt_positions_all = self.gt_positions

        for reconstruction in self.all_reconstructions:
            # Since the reconstruction might not only have points with a ground truth such as beads, we have to extract only the points which have a corresponding ground truth point
            matching_indexes = self.gt_positions.index.intersection(reconstruction.index)
            recon_with_gt = reconstruction.loc[matching_indexes]
            gt_with_recon = gt_positions_all.loc[matching_indexes]

            gt_positions = gt_with_recon.values
            reconstructed_positions_with_gt = recon_with_gt.values
            all_reconstructed_positions = reconstruction.values
            
            aligned_points = reconstruction.copy()
            # Ensure both reconstrcted and ground truth points are centered by removing the mean
            mean_gt = np.mean(gt_positions, axis=0)
            mean_recon = np.mean(reconstructed_positions_with_gt, axis=0)
            centered_gt = gt_positions - mean_gt
            centered_reconstructed = reconstructed_positions_with_gt - mean_recon
            
            # Perform Singular Value Decomposition (SVD) to find the rotation matrix
            U, S, Vt = np.linalg.svd(np.dot(centered_reconstructed.T, centered_gt))

            #correct for scaling by taking the norms of both points clouds
            norm_gt = np.linalg.norm(centered_gt)
            norm_recon = np.linalg.norm(centered_reconstructed)
            scale = norm_gt / norm_recon
            
            #assemble the transformation matrix
            A = np.dot(Vt.T, U.T)*scale

            #Notably, this transforms all points using  not only the ground truth points, based on the ground truth points and their reconstructed equivalent
            estimated_original_points = np.dot((all_reconstructed_positions - mean_recon), A.T) + mean_gt

            aligned_points.iloc[:, :] = estimated_original_points
            all_aligned_reconstructions.append(aligned_points)


        self.all_aligned_reconstructions = all_aligned_reconstructions # all aligned reconstructions are saved in this property as a list

    def calculated_morphed_reconstructions(self):
        list_morphed_reconstructions = [] #Create a list, all reconstructions follows the same structure of lists of dfs
        print("calculating morphed reconstruction")
        total = len(self.all_reconstructions)
        alpha = 0
        # reconstruction = self.all_reconstructions[0]
        for i, reconstruction in enumerate(self.all_reconstructions):

            percent = (i / total) * 100  # Calculate percentage
            bar = "#" * (i) + "-" * ((total) - (i))  # Create bar
            print(f"\r[{bar}] {percent:.1f}%", end="", flush=True)  # Print on the same line

            morphed_recon = reconstruction.copy()
            points = morphed_recon.values
            centroid = points.mean(axis=0)    # Mean for each column (x, y, and possibly z)
            radius = np.max(np.linalg.norm(points - centroid, axis = 1))
            normalized_points = (points - centroid)/radius
            # try:
            morphed_normalized_points  = alphamorph_apply(normalized_points, alpha=2, pca_mode= False) # morph the normalized points, the alpha is customizable
            # morphed_normalized_points_no_pca  = alphamorph_apply(normalized_points, alpha=2, pca_mode = False) # morph the normalized points, the alpha is customizable
            # except:
            #     morphed_normalized_points = normalized_points
            morphed_reconstruction_points = morphed_normalized_points*radius + centroid # un-normalize the points
            morphed_recon.loc[:,:] = morphed_reconstruction_points # re-add the numpy matrix points to the df
            list_morphed_reconstructions.append(morphed_recon) 
            # fig, (ax, ax2) = plt.subplots(1, 2, figsize = (12, 6))
            # gt_points = morphed_reconstruction_points
            # recon_points =morphed_normalized_points
            # print(gt_points)
            # print(recon_points)
            # from matplotlib.patches import Circle

            # center = (0,0)
            # circle = Circle(center, radius, alpha = 0.25, edgecolor = "k")
            # ax.add_patch(circle)
            # ax.set_aspect("equal")
            # ax.scatter(gt_points[:, 0], gt_points[:, 1], s = 1)
            # ax2.scatter(recon_points[:, 0], recon_points[:, 1], s = 1)
            # circle = Circle(center, radius, alpha = 0.25, edgecolor = "k")
            # ax2.add_patch(circle)
            # ax2.set_aspect("equal")
            # break
            # alpha +=0.2
            # break

        # plt.show()
        bar = "#" * ((total))  # Create bar
        print(f"\r[{bar}] {100.0:.1f}%", end="", flush=True)  # Print on the same line
        print()
        self.all_unaligned_morphed_reconstructions = list_morphed_reconstructions
        # self.all_aligned_morphed_reconstructions = list_morphed_reconstructions

    def optimize_reconstruction_alignment(self, type = "base"):
        '''
        We also re-align the morphed points to the ground truth, mainly to calculate the distortion, since the point positions should not change enough to be visually poor. 
        '''
        for type in ["morph", "base"]:
            all_aligned_reconstructions = []
            gt_positions_all = self.gt_positions
            from scipy.stats import linregress

            if type == "base":
                recons_to_align = self.all_reconstructions
            elif type =="morph":
                recons_to_align = self.all_unaligned_morphed_reconstructions
            list_distortions = []
            
            for reconstruction in recons_to_align:
                # Since the reconstruction might not only have points with a ground truth such as beads, we have to extract only the points which have a corresponding ground truth point
                matching_indexes = self.gt_positions.index.intersection(reconstruction.index)
                recon_with_gt = reconstruction.loc[matching_indexes]
                gt_with_recon = gt_positions_all.loc[matching_indexes]

                gt_positions = gt_with_recon.values
                reconstructed_positions_with_gt = recon_with_gt.values
                all_reconstructed_positions = reconstruction.values
                
                aligned_points = reconstruction.copy()

                methods = [0,1]
                # radii = [0.99]
                radii = np.arange(0.8, 1.3+0.01, 0.01)
                # radii = [1]
                best_method = None
                best_radius = None
                best_distortion = 100000
                for method in methods:
                    for radius in radii:
                        
                        if method == 0:
                            id = "simple mean"
                            centroid_gt = gt_positions.mean(axis=0)
                            centroid_recon = all_reconstructed_positions.mean(axis=0)
                            # min1, max1 = centered_gt[hull_gt.vertices].min(axis=0), centered_gt[hull_gt.vertices].max(axis=0)
                            # min2, max2 = centered_reconstructed[hull_recon.vertices].min(axis=0), centered_reconstructed[hull_recon.vertices].max(axis=0)
                            # print(min1, max1)
                            # quit()
                            radius_gt = np.max(np.linalg.norm(gt_positions - centroid_gt, axis = 1))*radius
                            radius_recon = np.max(np.linalg.norm(all_reconstructed_positions - centroid_recon, axis = 1))
                        elif method == 1:
                            id = "convex_hull_mean"
                            from scipy.spatial import ConvexHull
                            hull_gt = ConvexHull(gt_positions)
                            hull_vertices_gt = gt_positions[hull_gt.vertices]
                            centroid_gt = hull_vertices_gt.mean(axis=0)
                            hull_recon = ConvexHull(reconstructed_positions_with_gt)
                            hull_vertices_recon = reconstructed_positions_with_gt[hull_recon.vertices]
                            centroid_recon = hull_vertices_recon.mean(axis=0)
                            
                            radius_gt = np.max(np.linalg.norm(gt_positions - centroid_gt, axis = 1))*radius
                            radius_recon = np.max(np.linalg.norm(reconstructed_positions_with_gt - centroid_recon, axis = 1))

                        elif method == 2:
                            id = "geometric_median"
                            def geometric_median(points):
                                from scipy.optimize import minimize
                                import numpy as np
                                def aggregate_distance(x):
                                    return np.sum(np.linalg.norm(points - x, axis=1))
                                
                                centroid = np.mean(points, axis=0)
                                result = minimize(aggregate_distance, centroid)
                                return result.x

                            centroid_gt = geometric_median(gt_positions)
                            centroid_recon = geometric_median(reconstructed_positions_with_gt)
                            radius_gt = np.max(np.linalg.norm(gt_positions - centroid_gt, axis = 1))
                            radius_recon = np.max(np.linalg.norm(reconstructed_positions_with_gt - centroid_recon, axis = 1))
                        elif method == 3:
                            centroid_gt = geometric_median(gt_positions)
                            centroid_recon = geometric_median(reconstructed_positions_with_gt)
                            radius_gt = np.max(np.linalg.norm(gt_positions - centroid_gt, axis = 1))
                            radius_recon = np.max(np.linalg.norm(reconstructed_positions_with_gt - centroid_recon, axis = 1))
                        else:
                            continue

                        # print(id)

                        centered_gt = (gt_positions - centroid_gt)/radius_gt
                        centered_reconstructed = (reconstructed_positions_with_gt - centroid_recon)/radius_recon
                        centered_reconstructed_full = (all_reconstructed_positions - centroid_recon)/radius_recon
                        if method == 1:
                            new_recon_centroid = centered_reconstructed.mean(axis=0)
                            new_gt_centroid = centered_gt.mean(axis=0)
                            # print(new_recon_centroid, new_gt_centroid)
                            
                            min1, max1 = centered_gt[hull_gt.vertices].min(axis=0), centered_gt[hull_gt.vertices].max(axis=0)
                            min2, max2 = centered_reconstructed[hull_recon.vertices].min(axis=0), centered_reconstructed[hull_recon.vertices].max(axis=0)
                            center_gt = (min1 + max1)/2
                            center_recon = (min2 + max2)/2
                            # print(center_gt)
                            # print(center_recon)
                            # Compute translation vector to align spans (without scaling)
                            translation_vector_gt = -center_gt  # Shift GT to (0,0)
                            translation_vector_recon = -center_recon  # Shift reconstructed to (0,0)
                            # translation_vector_gt = [0,0]  # Shift GT to (0,0)
                            # translation_vector_recon = [0,0]  # Shift reconstructed to (0,0)
                            # Apply translation to align both
                            centered_gt = centered_gt + translation_vector_gt
                            centered_reconstructed = centered_reconstructed + translation_vector_recon
                            centered_reconstructed_full = centered_reconstructed_full + translation_vector_recon
                        

                        from scipy.spatial import ConvexHull
                        hull_gt = ConvexHull(centered_gt)
                        hull_vertices_gt = centered_gt[hull_gt.vertices]


                        
                        distances = np.linalg.norm(hull_vertices_gt - centroid_gt, axis=1)
                        # radius_gt = distances.mean()

                        hull_recon = ConvexHull(centered_reconstructed)
                        hull_vertices_recon = centered_reconstructed[hull_recon.vertices]

                        from scipy.spatial.distance import directed_hausdorff

                        # Use only the hull vertices
                        d1 = directed_hausdorff(hull_vertices_gt, hull_vertices_recon)[0]
                        d2 = directed_hausdorff(hull_vertices_recon, hull_vertices_gt)[0]
                        hausdorff_distance = max(d1, d2)

                        from scipy.spatial import cKDTree

                        tree_recon = cKDTree(hull_vertices_recon)
                        dists_gt_to_recon, _ = tree_recon.query(hull_vertices_gt)
                        avg_distance_gt_to_recon = dists_gt_to_recon.mean()

                        tree_gt = cKDTree(hull_vertices_gt)
                        dists_recon_to_gt, _ = tree_gt.query(hull_vertices_recon)
                        avg_distance_recon_to_gt = dists_recon_to_gt.mean()

                        mean_bidirectional_distance = (avg_distance_gt_to_recon + avg_distance_recon_to_gt) / 2

                        # print(f"Mean bidirectional hull distance: {mean_bidirectional_distance}")

                        area_gt = hull_gt.area
                        area_recon = hull_recon.area
                        area_ratio = area_recon / area_gt

                        # print(f"Area GT: {area_gt}, Area Recon: {area_recon}, Ratio: {area_ratio}")

                        from shapely.geometry import Polygon

                        poly_gt = Polygon(hull_vertices_gt)
                        poly_recon = Polygon(hull_vertices_recon)
                        intersection_area = poly_gt.intersection(poly_recon).area
                        union_area = poly_gt.union(poly_recon).area

                        iou = intersection_area / union_area
                        # print(f"Intersection over Union (IoU): {iou}")
                        if np.abs(area_ratio -1) >0.0:
                            # continue
                            # fig, ax = plt.subplots(1,1,figsize =(8,8))
                            # for simplex in hull_gt.simplices:
                            #     ax.plot(centered_gt[simplex, 0], centered_gt[simplex, 1], 'b-')
                            # ax.scatter(centered_gt[:,0], centered_gt[:,1], alpha = 0.5, s = 5, label = f"{centroid_gt[0]:.1f},{centroid_gt[1]:.1f}")
                            # ax.scatter(centered_reconstructed[:,0], centered_reconstructed[:,1], alpha = 0.5, s = 5, label = f"{centroid_recon[0]:.1f},{centroid_recon[1]:.1f}")
                            # ax.legend()
                            # ax.set_aspect("equal")
                            # ax.set_xlim([-1.05, 1.05])
                            # ax.set_ylim([-1.05, 1.05])

                            # ax.set_aspect("equal")
                            # for simplex in hull_recon.simplices:
                            #     ax.plot(centered_reconstructed[simplex, 0], centered_reconstructed[simplex, 1], 'r-')
                            # ax.set_title(f"Method {method} {id}, {radius}\n MBD:{mean_bidirectional_distance:.2f}, IOU: {iou:.2f}, haus D: {hausdorff_distance:.2f}, areas: {area_gt:.2f}/{area_recon:.2f} = {area_ratio:.2f}")
                            pass
                            # break
                        # ax.scatter(centroid_gt[0], centroid_gt[1], c = "m")
                        # ax.scatter(centered_reconstructed[0], centroid_recon[1], c = "c")

                        # plt.scatter(centered_gt[])

                        # Perform Singular Value Decomposition (SVD) to find the rotation matrix
                        U, S, Vt = np.linalg.svd(np.dot(centered_reconstructed.T, centered_gt))

                        #correct for scaling by taking the norms of both points clouds
                        # norm_gt = np.linalg.norm(centered_gt)
                        # norm_recon = np.linalg.norm(centered_reconstructed)
                        A = np.dot(Vt.T, U.T)
                        scale_estimation_recon_points = np.dot((reconstructed_positions_with_gt), A.T)

                        
                        #assemble the transformation matrix
                        A = np.dot(Vt.T, U.T)

                        #Notably, this transforms all points using  not only the ground truth points, based on the ground truth points and their reconstructed equivalent
                        slope, intercept, r_value, p_value, std_err = linregress(scale_estimation_recon_points.flatten(), gt_positions.flatten())
                        scale = radius_gt
                        if method==1:
                            estimated_original_points = (np.dot((centered_reconstructed_full), A.T)-translation_vector_gt)*scale + centroid_gt
                            estimated_original_points_gt = (np.dot((centered_reconstructed), A.T)-translation_vector_gt)*scale + centroid_gt
                        else:
                            estimated_original_points = (np.dot((centered_reconstructed_full), A.T))*scale + centroid_gt
                            estimated_original_points_gt = (np.dot((centered_reconstructed), A.T))*scale + centroid_gt
                        # estimated_original_points = (np.dot((centered_reconstructed_full), A.T))*scale + centroid_gt
                        # estimated_original_points_gt = (np.dot((centered_reconstructed), A.T))*scale + centroid_gt

                        estimated_original_points_gt_pre_rotation_mirror = np.dot((centered_reconstructed), A.T)

                        # fig, ax = plt.subplots(figsize=(7, 7))
                        # gt_points = centered_gt
                        # recon_points =centered_reconstructed
                        # print(gt_points)
                        # print()

                        # ax.set_aspect('equal')
                        # ax.set_box_aspect(1)
                        
                        # # ax.scatter(estimated_original_points_gt[:, 0], estimated_original_points_gt[:, 1], s = 3, alpha =0.3, c = "r")
                        # ax.scatter(gt_points[:, 0], gt_points[:, 1], s = 5, alpha = 0.3, c ="b")
                        # ax.scatter(recon_points[:, 0], recon_points[:, 1], s = 3, alpha =0.3, c ="g")
                        # def plot_hull(ax, points, color, label):
                        #     if len(points) < 3:  # ConvexHull requires at least 3 points
                        #         return
                            
                        #     hull = ConvexHull(points)
                        #     hull_points = np.append(hull.vertices, hull.vertices[0])  # Close the loop
                        #     ax.plot(points[hull_points, 0], points[hull_points, 1], color=color, linewidth=1.5, label=label)

                        # # Draw convex hulls for each point set
                        # plot_hull(ax, recon_points, "g", "Reconstruction Hull")
                        # # plot_hull(ax, estimated_original_points_basic, "r", "Estimated Original Hull")
                        # plot_hull(ax, gt_points, "b", "Ground Truth Hull")
                        
                        gt_points = gt_positions
                        recon_points =estimated_original_points_gt
                        distances = np.sqrt((gt_points[:, 0] - recon_points[:, 0]) ** 2 + (gt_points[:, 1] - recon_points[:, 1]) ** 2)
                        if np.median(distances) < best_distortion:
                            best_distortion = np.median(distances)
                            best_method = method
                            best_radius = radius
                            aligned_points.iloc[:, :] = estimated_original_points
                            best_reconstruction = recon_points
                            best_type = type

                            # print(np.median(distances), "mean_distortion aligned", radius, method)
                            # print(gt_points)
                            # print(best_distortion)
                # fig, ax = plt.subplots()
                # ax.set_aspect('equal')
                # list_distortions.append(best_distortion)
                # # ax.scatter(estimated_original_points_gt[:, 0], estimated_original_points_gt[:, 1], s = 3, alpha =0.3, c = "r")
                # ax.scatter(gt_points[:, 0], gt_points[:, 1], s = 5, alpha = 0.3, c ="b")
                # ax.scatter(best_reconstruction[:, 0], best_reconstruction[:, 1], s = 3, alpha =0.3, c ="g")
                # ax.set_title(f"{best_distortion} mean_distortion aligned, {best_radius:.2f}, {best_method}, {type}")
                # def plot_hull(ax, points, color, label):
                #     if len(points) < 3:  # ConvexHull requires at least 3 points
                #         return
                    
                #     hull = ConvexHull(points)
                #     hull_points = np.append(hull.vertices, hull.vertices[0])  # Close the loop
                #     ax.plot(points[hull_points, 0], points[hull_points, 1], color=color, linewidth=1.5, label=label)

                # # Draw convex hulls for each point set
                # plot_hull(ax, best_reconstruction, "g", "Reconstruction Hull")
                # # plot_hull(ax, estimated_original_points_basic, "r", "Estimated Original Hull")
                # plot_hull(ax, gt_points, "b", "Ground Truth Hull")
                all_aligned_reconstructions.append(aligned_points)    
            if type == "base":
                print(list_distortions, type)
                self.all_aligned_reconstructions = all_aligned_reconstructions # all aligned reconstructions are saved in this property as a list
            elif type == "morph":
                print(list_distortions, type)
                self.all_aligned_morphed_reconstructions = all_aligned_reconstructions # all aligned reconstructions are saved in this property as a list
        # print(best_distortion, best_method, best_radius, type)
        print()
        self.calculate_distortion()
        self.calculate_distortion(type = "morphed")
        print(self.median_dstrn_all_morphed_reconstructions, np.mean(self.median_dstrn_all_morphed_reconstructions)) 
        print(self.median_dstrn_all_reconstructions, np.mean(self.median_dstrn_all_reconstructions))
        # plt.show() 
        # quit()

    def calculate_knn(self, k = None, type ="standard"):
        from scipy.spatial import KDTree
        if k == None:
            k = self.config.subgraph_to_analyse.knn_neighbours

        if type =="standard":
            self.knn_all_reconstructions, self.mean_knn_all_reconstructions, self.median_knn_all_reconstructions, self.std_knn_all_reconstructions = [], [], [], []
            for reconstruction in self.all_reconstructions:
                # Since the reconstruction might not only have points with a ground truth such as beads, we have to extract only the points which have a corresponding ground truth point
                matching_indexes = self.gt_positions.index.intersection(reconstruction.index)
                gt_for_knn = self.gt_positions.loc[matching_indexes]
                recon_with_gt = reconstruction.loc[matching_indexes]

                original_tree = KDTree(gt_for_knn)  # construct the KD tree, 
                original_neighbors = original_tree.query(gt_for_knn, k + 1)[1][:, 1:] # k+1 since it counts itself, otherwise we would only get k-1 neighbours

                reconstructed_tree = KDTree(recon_with_gt)
                reconstructed_neighbors = reconstructed_tree.query(recon_with_gt, k + 1)[1][:, 1:]
                knn_per_point = []
                for original, reconstructed in zip(original_neighbors, reconstructed_neighbors):  # each row will be a unique cell, so we loop over them to acquire the ratio of shared points betweek the two             
                    n = len(original)
                    knn_per_point.append(len(set(original).intersection(set(reconstructed[:n]))) / n)
                
                self.knn_all_reconstructions.append(knn_per_point)
                self.mean_knn_all_reconstructions.append(np.mean(knn_per_point))
                self.median_knn_all_reconstructions.append(np.median(knn_per_point))    
                self.std_knn_all_reconstructions.append(np.std(knn_per_point))
            
        elif type =="morphed":
            #We have cases since the storage variables differ when we do it for morphed and normal
            self.knn_all_morphed_reconstructions, self.mean_knn_all_morphed_reconstructions, self.median_knn_all_morphed_reconstructions, self.std_knn_all_morphed_reconstructions = [], [], [], []
            for reconstruction in self.all_unaligned_morphed_reconstructions:
                # Since the reconstruction might not only have points with a ground truth such as beads, we have to extract only the points which have a corresponding ground truth point
                matching_indexes = self.gt_positions.index.intersection(reconstruction.index)
                gt_for_knn = self.gt_positions.loc[matching_indexes]
                recon_with_gt = reconstruction.loc[matching_indexes]

                original_tree = KDTree(gt_for_knn)
                original_neighbors = original_tree.query(gt_for_knn, k + 1)[1][:, 1:]

                reconstructed_tree = KDTree(recon_with_gt)
                reconstructed_neighbors = reconstructed_tree.query(recon_with_gt, k + 1)[1][:, 1:]
                knn_per_point = []
                for original, reconstructed in zip(original_neighbors, reconstructed_neighbors):               
                    n = len(original)
                    knn_per_point.append(len(set(original).intersection(set(reconstructed[:n]))) / n)
                
                self.knn_all_morphed_reconstructions.append(knn_per_point)
                self.mean_knn_all_morphed_reconstructions.append(np.mean(knn_per_point))
                self.median_knn_all_morphed_reconstructions.append(np.median(knn_per_point))    
                self.std_knn_all_morphed_reconstructions.append(np.std(knn_per_point))

    def calculate_cpd(self, point_limit=100000, type ="standard"):
        num_points = len(self.gt_positions)
        
        original_points = self.gt_positions
        # If desired, you can set an upper limit to how many points to calculate if more speed is desired at the cost of a bit of accuracy 
        # note that none of the samples have more than 10 000 ground truth points (cells) and therefore the default settings does nothing since it only counts the ground truth positions
        if num_points > point_limit: 
            random_indices = np.random.choice(num_points, point_limit, replace=False)
            original_points = original_points.iloc[random_indices]
        if type =="standard":
            self.cpd_all_reconstructions, self.reconstructed_pairwise_distances = [], []
            for reconstruction in self.all_reconstructions:
                # Since the reconstruction might not only have points with a ground truth such as beads, we have to extract only the points which have a corresponding ground truth point
                matching_indexes = self.gt_positions.index.intersection(reconstruction.index)
                recon_with_gt = reconstruction.loc[matching_indexes]
                gt_with_recon = original_points.loc[matching_indexes]

                original_distances = pdist(gt_with_recon) #Calculate all pairwise distances for the ground truth positions
                self.gt_pairwise_distances = original_distances
                if num_points > point_limit:
                    reconstruction = reconstruction.iloc[random_indices]
                reconstructed_distances = pdist(recon_with_gt)#Calculate all pairwise distances for the all reconstructed positions with a ground truth
                self.reconstructed_pairwise_distances.append(reconstructed_distances)
                if len(reconstructed_distances)<3: # if there are only two points, correlation will always be 1 otherwise
                    correlation = 0
                else:
                    correlation, _ = pearsonr(original_distances, reconstructed_distances) 
                r_squared = correlation**2
                self.cpd_all_reconstructions.append(r_squared)
                print(r_squared)

        elif type =="morphed":
            #We have cases since the storage variables differ when we do it for morphed and normal
            self.cpd_all_morphed_reconstructions, self.morphed_reconstructed_pairwise_distances = [], []
            for reconstruction in self.all_unaligned_morphed_reconstructions:

                matching_indexes = self.gt_positions.index.intersection(reconstruction.index)
                recon_with_gt = reconstruction.loc[matching_indexes]
                gt_with_recon = original_points.loc[matching_indexes]

                original_distances = pdist(gt_with_recon)
                self.gt_pairwise_distances = original_distances
                if num_points > point_limit:
                    reconstruction = reconstruction.iloc[random_indices]
                reconstructed_distances = pdist(recon_with_gt)
                self.morphed_reconstructed_pairwise_distances.append(reconstructed_distances)
                if len(reconstructed_distances)<3:
                    correlation = 0
                else:
                    correlation, _ = pearsonr(original_distances, reconstructed_distances)
                r_squared = correlation**2
                self.cpd_all_morphed_reconstructions.append(r_squared)
                print(r_squared)
        if num_points >1: # If we only have one point, there should be no "longest distance" and np.max cannot handle it
            self.longest_gt_distance = np.max(original_distances)
                   
    def calculate_distortion(self, type ="standard", read_file = False, quality_df = None):
        '''
        This functions calculated the distortion AKA the shift in position for each point from ground truth to reconstruction when the reconstruction is aligned to the ground truth.
        As a metric this suffers from compounding issues in that not only will the reconstruction result in some shift due to loss of exact spatial information,
        the alignment to ground truth also introduces some shifts due to working towards the best fit for all points. This means that although certain regions could potentially be perfectly reconstructed, 
        The alignments global goal will still shift them from position.
        This makes metrics such as KNN or CPD potentially more accurate metrics to assess the reconstructions quality
        '''
        # if self.all_aligned_reconstructions ==None: # reconstructions has not been aligned to the ground truth, it does that. This should not happen with normal usage
        #     self.align_reconstructions_to_gt_svd()

        if type =="standard":
            self.dstrn_all_reconstructions, self.mean_dstrn_all_reconstructions, self.median_dstrn_all_reconstructions, self.std_dstrn_all_reconstructions = [], [], [], []
            for aligned_reconstruction in self.all_aligned_reconstructions:
                # The distortion itself is simply the euclidean distance between reconstruction and ground truth when the reconstruction has been aligned to the ground truth
                try: #The detailed reconstructions summary uses another column formatting, this accounts for that since it is used for both plotting and making files
                    distances = np.sqrt((self.gt_positions['x'] - aligned_reconstruction['x']) ** 2 + (self.gt_positions['y'] - aligned_reconstruction['y']) ** 2)
                except:
                    distances = np.sqrt((self.gt_positions['gt_x'] - aligned_reconstruction['x']) ** 2 + (self.gt_positions['gt_y'] - aligned_reconstruction['y']) ** 2)                
                distances_df = pd.DataFrame(distances, columns=['distance'], index=self.gt_positions.index)

                self.dstrn_all_reconstructions.append(distances_df)
                self.mean_dstrn_all_reconstructions.append(np.mean(distances_df.values))
                self.median_dstrn_all_reconstructions.append(np.median(distances_df.values))
                self.std_dstrn_all_reconstructions.append(np.std(distances_df.values))

        elif type =="morphed":
            #We have cases since the storage variables differ when we do it for morphed and normal
            self.dstrn_all_morphed_reconstructions, self.mean_dstrn_all_morphed_reconstructions, self.median_dstrn_all_morphed_reconstructions, self.std_dstrn_all_morphed_reconstructions = [], [], [], []
            for aligned_reconstruction in self.all_aligned_morphed_reconstructions:
                try: #The detailed reconstructions summary uses another column formatting, this accounts for that
                    distances = np.sqrt((self.gt_positions['x'] - aligned_reconstruction['x']) ** 2 + (self.gt_positions['y'] - aligned_reconstruction['y']) ** 2)
                except:
                    distances = np.sqrt((self.gt_positions['gt_x'] - aligned_reconstruction['x']) ** 2 + (self.gt_positions['gt_y'] - aligned_reconstruction['y']) ** 2)                
                distances_df = pd.DataFrame(distances, columns=['distance'], index=self.gt_positions.index)

                self.dstrn_all_morphed_reconstructions.append(distances_df)
                self.mean_dstrn_all_morphed_reconstructions.append(np.mean(distances_df.values))
                self.median_dstrn_all_morphed_reconstructions.append(np.median(distances_df.values))
                self.std_dstrn_all_morphed_reconstructions.append(np.std(distances_df.values))
            pass

    #region plotting and plotting related functions
    def initialize_figure(self, type = None, n_reconstructions = 2):
        '''
        This functions creates the figure window, depending on how many reconstruction and subgraph modfication we have as dictated by the config
        '''
        if type =="enrichment":
            
            ax_gt_space = n_reconstructions

            if n_reconstructions ==1 and (self.unenriched_flag or not self.config.subgraph_to_analyse.include_ungated) and self.config.subgraph_to_analyse.gating_threshold !="all": # This clause make a simplified and more visually appealing plot if no modified subgraphs are included
                fig= plt.figure(figsize=(16, 6))
                gs = fig.add_gridspec(1, 3)
                ax_gt_all = fig.add_subplot(gs[0,0])
                ax_gt_subgraph = fig.add_subplot(gs[0,1])
                fontsize = 10
            else:
                if ax_gt_space ==1:
                    ax_gt_space =2
                fig= plt.figure(figsize=(16, 3*ax_gt_space))
                gs = fig.add_gridspec(ax_gt_space+1, ax_gt_space+1+len(self.all_enrichedSubgraphs)) # The unmodified subgraph and its modified version all get their own window, and each reconstruction as well
                ax_gt_all = fig.add_subplot(gs[:ax_gt_space, :ax_gt_space])
                ax_gt_subgraph = fig.add_subplot(gs[0, ax_gt_space])
                
                fontsize = 100/(ax_gt_space+1+ ax_gt_space+1+len(self.all_enrichedSubgraphs))

            ax_gt_subgraph.set_xticks([]) # We remove the ticks of the subgraph plots, since its not as important and takes a lot of space
            ax_gt_subgraph.set_yticks([])
            ax_gt_subgraph.set_aspect("equal")
            return fig, gs, ax_gt_all, ax_gt_subgraph, fontsize, ax_gt_space
        
    def plot_ground_truth(self, ax = None, position = 10, clr_scheme=None, size = 8, with_edges = False):
        '''
        This functions plots the ground truth positions of the subgraph on any axis desired, and also uses the colourscheme specified in the config.
        Since we only need to plot nodes with a ground truth, it is relatively simpler than the reconstruction plotting function
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # Default to square plot if standalone

        if size == None:
            size = max([2e4/self.n_points_with_gt, 8])
        
        if clr_scheme == None:
            clr_scheme = self.config.vizualisation_args.color_scheme
        x_values, y_values, cell_types, colors = [], [], [], []

        # Due to the color schemes being based on ground truth positions, first we need to know which nodes in the subgraph has a ground truth
        barcodes_with_gt = [self.bc_to_idx[node] for node in self.point_ids if self.bc_to_idx[node] in self.config.gt_points.gt_df.index]
        gt_points = self.config.gt_points.gt_df.loc[barcodes_with_gt]

        x_values = gt_points["x"].values
        y_values = gt_points["y"].values
       
        cell_types = gt_points["cell_type"].values

        colormap = plt.get_cmap(self.config.vizualisation_args.colormap) # The colormap is choosable in the config file
        # try: # most colorschemes are quite self-explanatory
        if clr_scheme == "vertical":
            y_norm = Normalize(vmin=y_values.min(), vmax=y_values.max())(y_values)
            colors = colormap(y_norm)
        elif clr_scheme == "horizontal":
            x_norm = Normalize(vmin=x_values.min(), vmax=x_values.max())(x_values)
            colors = colormap(x_norm)
        elif clr_scheme == "radius":
            mean_x, mean_y = x_values.mean(), y_values.mean()
            distance_from_mean = np.sqrt((x_values - mean_x)**2 + (y_values - mean_y)**2)
            distance_norm = Normalize(vmin=distance_from_mean.min(), vmax=distance_from_mean.max())(distance_from_mean)
            colors = colormap(distance_norm)
        elif clr_scheme == "cell_type":
            colors = [self.colors[cell_type] for cell_type in cell_types]
        elif clr_scheme == "knn":
            norm = Normalize(vmin=0, vmax=1)
            colors = colormap(norm(self.current_knn))
        elif clr_scheme == "distortion":
            norm = Normalize(vmin=0, vmax=self.current_distortion.max())
            colors = colormap(norm(self.current_distortion))
        elif clr_scheme =="image":
            print(x_values)
            from PIL import Image
            image = Image.open("Input_files/DNA_black_and_white.png").convert("L")
            image_array = np.array(image)
            self.binary_image = image_array < 128  # black if pixel value < 128
            height, width = self.binary_image.shape

            # Normalize your point cloud to [0, 1] range
            # Example point cloud
            points = gt_points[["x", "y"]].values  # 100 points with x,y in [0, 1]
            self.factors_for_image_coloring = []
            self.factors_for_image_coloring.append(points[:, 0].min(axis=0))
            points[:, 0] = points[:, 0]-points[:, 0].min(axis=0)
            self.factors_for_image_coloring.append(points[:, 1].min(axis=0))
            points[:, 1] = points[:, 1]-points[:, 1].min(axis=0)
            self.factors_for_image_coloring.append(points.max())
            self.factors_for_image_coloring.append(width)
            self.factors_for_image_coloring.append(height)
            points /= points.max()
            # self.factors_for_image_coloring = [points[:, 0].min(axis=0), points[:, 1].min(axis=0), points.max(), width, height]
            print(points)
            # Map normalized points to image pixel indices
            pixel_x = (points[:, 0] * (width-1)).astype(int)
            pixel_y = (points[:, 1] * (height-1)).astype(int)
            print(height, width)
            print(pixel_x)
            # Get color classification based on pixel values
            colors = ['k' if self.binary_image[y, x] else 'w' for x, y in zip(pixel_x*-1, pixel_y)]

            # Plotting
            # plt.imshow(binary_image, cmap='gray', extent=[0, 1, 0, 1])
            # plt.scatter(points[:, 0], points[:, 1], c=colors, edgecolors='k')
            # plt.gca().set_aspect('equal')
            # plt.title("Points Colored by Image Region (Normalized)")
            # plt.xlabel("x (normalized)")
            # plt.ylabel("y (normalized)")
            # plt.show()
            # print(image_array)
            # print(clr_scheme)
            # quit()
        else: # default to cell-type coloring
            colors = [self.colors[cell_type] for cell_type in cell_types]
        # except:
        #     print("Defaulting to cell coloring in for ground truth") # This should not be needed, but acts as asafety catch
        #     colors = [self.colors[cell_type] for cell_type in cell_types]
        scalebar = AnchoredSizeBar(ax.transData,
                           500,                # length of the bar in data units
                           '500 units',        # label
                           'lower right',     # location
                           pad=0.5,
                           color='black',
                           frameon=False,
                           size_vertical=0.5,
                           fontproperties=fm.FontProperties(size=8))

        ax.add_artist(scalebar)
        ax.scatter(x_values, y_values, c = colors, zorder = position,  s=size, edgecolor = "k", linewidths =0.3)
        ax.set_aspect("equal")
        if with_edges:
            edges = self.generate_plottable_edges(self.gt_positions, self.original_edgelist)
            ax.add_collection(edges)
        return colors

    def plot_reconstruction(self, reconstruction, ax = None, position = 10, clr_scheme=None, size = 8, with_edges = False):
        '''
        This functions plots the reconstructed positions of the subgraph on any axis desired, for cells with a ground truth it uses the config specified color scheme.
        It is a bit mroe complex since it has to also plot nodes without a ground truth i.e. beads and non-ground truth cells
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # Default to square plot if standalone
        
        if size == None:
            size = max([2e4/self.n_points_with_gt, 4])
        
        if clr_scheme == None:
            clr_scheme = self.config.vizualisation_args.color_scheme
        # each type of data, data no gt should only be beads
        data_with_gt = {"x": [], "y": [], "z": [], "cell_type": [], "color": [], "size": []}
        data_no_gt = {"x": [], "y": [], "z": [], "color": [], "size": []}
        cells_no_gt = {"x": [], "y": [], "z": [], "color": [], "size": []}


        # Precompute for ground truth points, similarly to when we plot only them
        points_with_gt = [
            node for node in self.point_ids
            if self.bc_to_idx.get(node) in self.config.gt_points.gt_df.index
        ]
        barcodes_gt = [self.bc_to_idx[key] for key in points_with_gt]
        all_x_values_gt = self.config.gt_points.gt_df.loc[barcodes_gt, "x"]
        all_y_values_gt = self.config.gt_points.gt_df.loc[barcodes_gt, "y"]

        mean_x, mean_y = all_x_values_gt.mean(), all_y_values_gt.mean()
        distances_from_mean = np.sqrt((all_x_values_gt - mean_x) ** 2 + (all_y_values_gt - mean_y) ** 2)
        colormap = plt.get_cmap(self.config.vizualisation_args.colormap)

        # Normalize values for coloring of the positional modes, very quick so we do this irregardless
        y_norm = Normalize(vmin=all_y_values_gt.min(), vmax=all_y_values_gt.max())
        x_norm = Normalize(vmin=all_x_values_gt.min(), vmax=all_x_values_gt.max())
        radius_norm = Normalize(vmin=distances_from_mean.min(), vmax=distances_from_mean.max())
        cell_bc_length = len(barcodes_gt) # This is to identify cells that do not have a ground truth
        low_on_bottom = True # This just controls if points with a high value is specifically plotted on top or bottom
        # Loop through all nodes once
        for node in self.point_ids: # loop thorugh ALL points of the reconstruction, could potentially be vectorized if needed
            node_bc = self.bc_to_idx.get(node)
            reconstructed_node = reconstruction.loc[node]

            # Common properties
            x, y = reconstructed_node["x"], reconstructed_node["y"]
            z = reconstructed_node["z"] if self.reconstruction_dimension == 3 else None # If the reconstructino is 3D, we also have to include z
            # Check if node has ground truth, and color it based on the color scheme, in addition to appending its data to the correct dictionaries
            if node in points_with_gt:
                if len(node_bc)!=cell_bc_length: 
                    cell_bc_length = len(node_bc)
                gt_point = self.config.gt_points.gt_df.loc[node_bc]
                cell_type = gt_point["cell_type"]

                # Append data for nodes with GT
                data_with_gt["x"].append(x)
                data_with_gt["y"].append(y)
                data_with_gt["z"].append(z)
                data_with_gt["cell_type"].append(cell_type)
                data_with_gt["size"].append(size)

                # Determine color based on scheme
                if clr_scheme == "cell_type":
                    color = self.colors[cell_type]
                    color ="w"
                elif clr_scheme == "vertical":
                    low_on_bottom = False
                    color = colormap(y_norm(gt_point["y"]))
                elif clr_scheme == "horizontal":
                    color = colormap(x_norm(gt_point["x"]))
                elif clr_scheme == "radius":
                    distance_from_mean = np.sqrt((gt_point["x"] - mean_x) ** 2 + (gt_point["y"] - mean_y) ** 2)
                    color = colormap(radius_norm(distance_from_mean))
                elif clr_scheme == "image":
                    factors = self.factors_for_image_coloring
                    pixel_x = int(((gt_point["x"] - factors[0])/factors[2])*(factors[3]-1))*-1
                    pixel_y = int(((gt_point["y"] - factors[1])/factors[2])*(factors[4]-1))
                    if self.binary_image[pixel_y, pixel_x]:
                        color = "k"
                    else:
                        color = "w"
                    low_on_bottom=False
                    # Get color classification based on pixel values
                    # colors = ['k' if self.binary_image[y, x] else 'w' for x, y in zip(pixel_x*-1, pixel_y)]
                else:
                    color = "w"

                data_with_gt["color"].append(color)
            else:
                if "pseudo" in node_bc: # DBSCAN modification can result in "pseudocells" for cells with multiple clusters the not-biggest cluster will then be assigned pseudocells which this accounts for
                    cells_no_gt["x"].append(x)
                    cells_no_gt["y"].append(y)
                    cells_no_gt["z"].append(z)
                    
                    cells_no_gt["color"].append("w")

                    cells_no_gt["size"].append(size*0.5)
                elif len(node_bc)!=cell_bc_length:
                    # Append data for nodes with a barcode of a different length i.e beads
                    data_no_gt["x"].append(x)
                    data_no_gt["y"].append(y)
                    data_no_gt["z"].append(z)
                    data_no_gt["color"].append("w")
                    data_no_gt["size"].append(1)
                else: #This will be unknown cells
                    coloring = self.reconstruction_summary.copy().set_index("node_bc")
                    check_if_known_cells = coloring.loc[node_bc, "node_type"]

                    if check_if_known_cells != "unknown_cell":
                        cells_no_gt["color"].append(self.colors[check_if_known_cells])
                    else:
                        cells_no_gt["color"].append("gray")
                    cells_no_gt["x"].append(x)
                    cells_no_gt["y"].append(y)
                    cells_no_gt["z"].append(z)
                    # cells_no_gt["color"].append("gray")
                    cells_no_gt["size"].append(size*0.5)

        # Used for the titles
        self.n_cells = len(cells_no_gt["x"]) + len(data_with_gt["x"])
        self.n_gt_cells = len(data_with_gt["x"]) 
        self.n_all_nodes = len(cells_no_gt["x"]) + len(data_with_gt["x"]) + len(data_no_gt["x"])
        
        if clr_scheme == "knn":
            data_with_gt["color"] = self.current_knn
            low_on_bottom = False
        elif clr_scheme == "distortion":
            data_with_gt["color"] = self.current_distortion.values.flatten()
        
        if clr_scheme == "cell_type": # We do not want to sort the cells by their color assignment, so they are simply plotted by order of index, which are much more visually interpretable since the ground truth positiosn follow the same logic
            data_with_gt_df = pd.DataFrame(data_with_gt)
        else:
            data_with_gt_df = pd.DataFrame(data_with_gt).sort_values(by="color", ascending = low_on_bottom)
        
        if self.reconstruction_dimension ==3: # Include Z if needed
            if len(data_no_gt["x"])<=1e4:
                ax.scatter(data_no_gt["x"], data_no_gt["y"], data_no_gt["z"], c = data_no_gt["color"], zorder = position-1, s=data_no_gt["size"], edgecolor = "k", linewidths =0.1, alpha = 0.6)
            ax.scatter(cells_no_gt["x"], cells_no_gt["y"], cells_no_gt["z"], c = cells_no_gt["color"], zorder = position-1, s=cells_no_gt["size"], edgecolor = "k", linewidths =0.3)
            ax.scatter(data_with_gt["x"], data_with_gt["y"], data_with_gt["z"], c = data_with_gt["color"], s=np.array(data_with_gt["size"])*5, edgecolor = "k", linewidths =0.3, zorder = position-1)

        else:

            if len(data_no_gt["x"])<=1e4:
                # pass
                ax.scatter(data_no_gt["x"], data_no_gt["y"], c = data_no_gt["color"], zorder = position-1, s=data_no_gt["size"], edgecolor = "k", linewidths =0.1, alpha = 0.6)
            ax.scatter(cells_no_gt["x"], cells_no_gt["y"], c = cells_no_gt["color"], zorder = position-1, s=cells_no_gt["size"], edgecolor = "k", linewidths =0.1)
            ax.scatter(data_with_gt_df["x"], data_with_gt_df["y"], c = data_with_gt_df["color"], zorder = position, s=data_with_gt_df["size"], edgecolor = "k", linewidths =0.3)
        
        scalebar = AnchoredSizeBar(ax.transData,
                           500,                # length of the bar in data units
                           '500 units',        # label
                           'lower right',     # location
                           pad=0.5,
                           color='black',
                           frameon=False,
                           size_vertical=0.5,
                           fontproperties=fm.FontProperties(size=8))

        ax.add_artist(scalebar)
        if with_edges: # edges are generated with a separate functions in the form of a LineCollection
            edges = self.generate_plottable_edges(self.gt_positions, self.original_edgelist)
            ax.add_collection(edges)

    def generate_plottable_edges(self, points, edges, all_have_positions = False, dimension=2):

        all_edges = []
        # The format required for usignt he LineCollection is essentially a 3D matrix, where each row is a points and with the z column points 1 = z1 and point 2 = z2
        if dimension == 2: # 3D lines require another collection
            from matplotlib.collections import LineCollection
            if len(edges)>100000 and not self.config.vizualisation_args.include_edges:
                print("Number of edges exceed reasonable plotting capacities") # The point format of the edges is quite tedious to create, and also plot so if there are too many it skips it unless specifically told to plot them by config
                return LineCollection([])
            # source_positions = points.loc[edges["source"], ["x", "y"]].to_numpy()
            # target_positions = points.loc[edges["target"], ["x", "y"]].to_numpy()

            # # Stack the positions directly to create the edge list
            # all_edges = np.stack([source_positions, target_positions], axis=1).tolist()
            print(edges)
            for idx, (source, target) in edges[["source", "target"]].iterrows():
                all_edges.append([[points.loc[source]["x"], points.loc[source]["y"]], [points.loc[target]["x"], points.loc[target]["y"]]])
                self.edges_matrix_format = all_edges
            return LineCollection(all_edges, color = "gray", linewidth = 1, alpha = 0.3)
        
        elif dimension == 3:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            if len(edges)>100000 and not self.config.vizualisation_args.include_edges:
                print("Number of edges exceed reasonable plotting capacities")
                return False
            for idx, (source, target) in edges.iterrows():
                edge = [
                [points.loc[source]["x"], points.loc[source]["y"], points.loc[source]["z"]],
                [points.loc[target]["x"], points.loc[target]["y"], points.loc[target]["z"]]
                    ]
                all_edges.append(edge)
            self.edges_matrix_format = all_edges
            return Line3DCollection(all_edges, colors="gray", linewidths = 0.1, alpha=0.1)
        
    def plot_distortion(self, aligned_reconstruction, ax = None, subsample = True, subsample_n = 100):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # Default to square plot if standalone
        from matplotlib.collections import LineCollection
        from matplotlib import colors as mcolors

        if subsample:
            true_positions = self.gt_positions.iloc[:subsample_n, :]
        else:
            true_positions = self.gt_positions.copy()

        true_positions.columns = ["gt_x", "gt_y"]
        common_indices = aligned_reconstruction.index.intersection(true_positions.index)
        reconstructed_positions = aligned_reconstruction.loc[common_indices]
        true_positions = true_positions.loc[common_indices]

        reconstructed_positions.columns = ["recon_x", "recon_y"]
        gt_recon_positions_df = pd.concat([true_positions, reconstructed_positions], axis=1)
        gt_recon_positions_df["distance"] = np.sqrt((gt_recon_positions_df['gt_x'] - gt_recon_positions_df['recon_x']) ** 2 + (gt_recon_positions_df['gt_y'] - gt_recon_positions_df['recon_y']) ** 2)

        norm = mcolors.Normalize(vmin=0, vmax=500) #gt_recon_positions_df["distance"].min()
        cmap = plt.get_cmap(f"{self.config.vizualisation_args.colormap}")

        colors = cmap(norm(gt_recon_positions_df["distance"]))
        gt_recon_positions_df.sort_values(by = "distance", ascending=True)
        distortion_lines = [
            [[original_x, original_y], [recon_x, recon_y]]
            for (idx, (original_x, original_y)), (idx, (recon_x, recon_y))
            in zip(true_positions.iterrows(), reconstructed_positions.iterrows())
        ]
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Required for the colorbar to work properly

        # Add the colorbar to the same axis
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Distance")
        distortion_lines = LineCollection(distortion_lines, colors=colors, linewidth=1, alpha=0.5)
        ax.add_collection(distortion_lines)
        ax.scatter(gt_recon_positions_df['gt_x'], gt_recon_positions_df['gt_y'], s =0, c = "gray")
        ax.scatter(gt_recon_positions_df['recon_x'], gt_recon_positions_df['recon_y'], s = 0, c = "k")

    def additional_initialization_and_saving_reconstruction_metrics(self, save_type = "reconstruction_quality_gt", calc_type = "standard"):
        '''
        This function creates and save the file for all reconstruction and also the file for the reconstruction metrics for any cells with a ground truth.
        It also calculates properties needed by the subgraph such as quality metrics, alignment and morphing
        '''
        quality_df_file_check = self.save_output_file(type = save_type, file_type = "quality_df", mode = "check") # The save file can also be used to check whether the file exists
        if quality_df_file_check and not self.config.subgraph_to_analyse.regenerate_summary_files: # If the file exists, do not recreate it unless specifically told so in the config
            self.quality_df = pd.read_csv(quality_df_file_check)
            # Even if the file exist, the quality metrics are also calculated in this function since all subgraphare initalized using it
            self.calculate_distortion(type = calc_type)
            self.calculate_knn(type = calc_type)
            self.calculate_cpd(type = calc_type)
            return

        quality_df = self.gt_positions.copy()
        if self.all_aligned_reconstructions ==None:
            print("Initializing reconstructions")
            # If the subgraph hasnt already found its reconstructions, it does so
            self.find_reconstructions()
        else:
            matching_indexes = quality_df.index.intersection(self.all_aligned_reconstructions[0].index)
            quality_df = quality_df.loc[matching_indexes]
        if self.reconstruction_dimension != 2:
            self.all_aligned_reconstructions = self.all_reconstructions
        self.calculate_distortion(type = calc_type)
        self.calculate_knn(type = calc_type)
        self.calculate_cpd(type = calc_type)
        
        quality_df.columns = ["gt_x", "gt_y"] # This df will be succesively built to create the full quality metrics df, that will be written to file
        for i, (recon, knn, distortion) in enumerate(zip(self.all_aligned_reconstructions, self.knn_all_reconstructions, self.dstrn_all_reconstructions)):
            
            quality_df[f"knn_{i+1}"] = knn
            quality_df[f"distortion_{i+1}"] = distortion
        if calc_type == "morphed": # additional columns for the morphed reconstructions
            for i, (recon, knn, distortion) in enumerate(zip(self.all_unaligned_morphed_reconstructions, self.knn_all_morphed_reconstructions, self.dstrn_all_morphed_reconstructions)):
                quality_df[f"morphed_knn_{i+1}"] = knn
                quality_df[f"morphed_distortion_{i+1}"] = distortion

        quality_df["mv"] = quality_df.index # This column is just used to shuffle coluumns around to a more intuitive order
        column_to_move = quality_df["mv"].map(self.bc_to_idx)  # This converts node idx's to barcodes instead
        quality_df.insert(0, "bc", column_to_move)  # Insert it at position 0 (far left)
        quality_df.insert(1, "cell_type", self.config.gt_points.gt_df.loc[quality_df["bc"], "cell_type"].values) # Insert the cell types as well

        quality_df.pop("mv")
        self.quality_df = quality_df
        self.save_output_file(quality_df, type = save_type, file_type = "quality_df")
    
    def create_and_save_full_reconstruction_summary(self, save_type ="full_reconstruction_summary", enriched = False):
        full_reconstruction_file_check = self.save_output_file(type = save_type, file_type = "full_reconstruction_summary", enriched=enriched, mode = "check")
        if full_reconstruction_file_check and not self.config.subgraph_to_analyse.regenerate_summary_files:
            self.reconstruction_summary = pd.read_csv(full_reconstruction_file_check)
            reconstruction_summary = self.reconstruction_summary
            recon_x = [col for col in reconstruction_summary.columns if col.startswith("recon_x_")]
            recon_y = [col for col in reconstruction_summary.columns if col.startswith("recon_y_")]
            self.all_reconstructions = []
            for recon_x, recon_y in zip(recon_x, recon_y):
                recon_df = pd.DataFrame({"x":reconstruction_summary[recon_x].values, "y":reconstruction_summary[recon_y].values, "node_ID": reconstruction_summary["node_ID"]}).set_index("node_ID")
                self.all_reconstructions.append(recon_df)
            print(reconstruction_summary)
            align_recon_x = [col for col in reconstruction_summary.columns if col.startswith("align_recon_x_")]
            align_recon_y = [col for col in reconstruction_summary.columns if col.startswith("align_recon_y_")]
            self.all_aligned_reconstructions = []
            for recon_x, recon_y in zip(align_recon_x, align_recon_y):
                recon_df = pd.DataFrame({"x":reconstruction_summary[recon_x].values, "y":reconstruction_summary[recon_y].values, "node_ID": reconstruction_summary["node_ID"]}).set_index("node_ID")
                self.all_aligned_reconstructions.append(recon_df)
            
            morphed_recon_x = [col for col in reconstruction_summary.columns if col.startswith("morph_recon_x_")]
            morphed_recon_y = [col for col in reconstruction_summary.columns if col.startswith("morph_recon_y_")]

            self.all_unaligned_morphed_reconstructions = []
            for recon_x, recon_y in zip(morphed_recon_x, morphed_recon_y):
                recon_df = pd.DataFrame({"x":reconstruction_summary[recon_x].values, "y":reconstruction_summary[recon_y].values, "node_ID": reconstruction_summary["node_ID"]}).set_index("node_ID")
                self.all_unaligned_morphed_reconstructions.append(recon_df)
            self.optimize_reconstruction_alignment(type = "morph")
            # print("huh")
            # quit()
            # align_morphed_recon_x = [col for col in reconstruction_summary.columns if col.startswith("align_morph_recon_x_")]
            # align_morphed_recon_y = [col for col in reconstruction_summary.columns if col.startswith("align_morph_recon_y_")]

            # self.all_aligned_morphed_reconstructions = []
            # for recon_x, recon_y in zip(align_morphed_recon_x, align_morphed_recon_y):
            #     recon_df = pd.DataFrame({"x":reconstruction_summary[recon_x].values, "y":reconstruction_summary[recon_y].values, "node_ID": reconstruction_summary["node_ID"]}).set_index("node_ID")
            #     self.all_aligned_morphed_reconstructions.append(recon_df)


        if self.all_reconstructions ==None:
            print("Initializing reconstructions")
            self.find_reconstructions()

        reconstruction_summary = self.all_reconstructions[0].copy()
        print(f"self.is_modified_subgraph {self.is_modified_subgraph}")
        
        if len(self.all_reconstructions[0].columns) ==2:
            
            reconstruction_summary.columns = [f"recon_x_{1}", f"recon_y_{1}"]
            self.align_reconstructions_to_gt_svd()
            if self.config.modification_type =="dbscan" and self.is_modified_subgraph:
                name_qa_file= f"reconstruction_quality_gt_subgraph_{self.number}_N={self.n_nodes}_dbscan_{self.full_parameters}"
            elif enriched: 
                name_qa_file= f"reconstruction_quality_gt_subgraph_{self.number}_N={self.n_nodes}_{self.config.modification_type}_{self.enrichment_threshold}"
            else:
                name_qa_file = f"reconstruction_quality_gt_subgraph_{self.number}_N={self.n_nodes}"
            print("Saving Reconstruction quality metrics")
            self.additional_initialization_and_saving_reconstruction_metrics(save_type=name_qa_file)
            if not self.all_unaligned_morphed_reconstructions:
                # self.all_morphed_reconstructions = self.all_aligned_reconstructions # for testing
                self.calculated_morphed_reconstructions()
                # self.all_unaligned_morphed_reconstructions = self.all_aligned_reconstructions
                self.optimize_reconstruction_alignment()
            print("Saving Morphed reconstruction quality metrics")
            self.additional_initialization_and_saving_reconstruction_metrics(save_type=name_qa_file, calc_type="morphed")
            if full_reconstruction_file_check and not self.config.subgraph_to_analyse.regenerate_summary_files:
                return
        elif len(self.all_reconstructions[0].columns) ==3:
            reconstruction_summary.columns = [f"recon_x_{1}", f"recon_y_{1}", f"recon_z_{1}"]
        print("Generating full reconstruction summary")
        for i, (recon, align_recon, morph_recon, align_morph_recon) in enumerate(zip(self.all_reconstructions, self.all_aligned_reconstructions, self.all_unaligned_morphed_reconstructions, self.all_aligned_morphed_reconstructions)):
            if len(recon.columns) ==2:
                reconstruction_summary[f"recon_x_{i+1}"] = recon["x"]
                reconstruction_summary[f"recon_y_{i+1}"] = recon["y"]
                reconstruction_summary[f"align_recon_x_{i+1}"] = align_recon["x"]
                reconstruction_summary[f"align_recon_y_{i+1}"] = align_recon["y"]
                reconstruction_summary[f"morph_recon_x_{i+1}"] = morph_recon["x"]
                reconstruction_summary[f"morph_recon_y_{i+1}"] = morph_recon["y"]
                reconstruction_summary[f"align_morph_recon_x_{i+1}"] = align_morph_recon["x"]
                reconstruction_summary[f"align_morph_recon_y_{i+1}"] = align_morph_recon["y"]
            elif len(recon.columns) ==3:
                reconstruction_summary[f"recon_x_{i+1}"] = recon["x"]
                reconstruction_summary[f"recon_y_{i+1}"] = recon["y"]
                reconstruction_summary[f"recon_z_{i+1}"] = recon["z"]
            else:
                for j in range(len(recon.columns)): #if we ever want reconstructions in >3D
                    reconstruction_summary[f"recon_dim{j}_{i+1}"] = recon["z"]

        reconstruction_summary.reset_index(inplace=True)
        reconstruction_summary["node_bc"] = reconstruction_summary["node_ID"].map(self.bc_to_idx)
        reconstruction_summary["type_prediction_score"] = reconstruction_summary["node_bc"].copy()
        
        
        reconstruction_summary["node_type"] = reconstruction_summary["node_bc"]
        reconstruction_summary.loc[
                # (detailed_edgelist[col].apply(len) == len(self.config.gt_points.gt_df.index[0])) & 
                # (~detailed_edgelist[col].str.contains("pseudo", na=False)),
            (reconstruction_summary["node_type"].apply(len) != len(self.config.gt_points.gt_df.index[0])) &
            (~reconstruction_summary["node_type"].str.contains("pseudo", na=False)),
            "node_type"
        ] = "bead"

        
        reconstruction_summary.loc[
            reconstruction_summary["node_type"].isin(self.config.gt_points.gt_df.index),
            "node_type"
        ] = reconstruction_summary["node_type"].map(self.config.gt_points.gt_df["cell_type"])

        reconstruction_summary.loc[
            reconstruction_summary["node_bc"].isin(self.predicted_cell_types.index),
            "node_type"
        ] = reconstruction_summary["node_type"].map(self.predicted_cell_types["node_type"])
        #Important to not switch back barcodes that has already had its cell type inserted, in case the cell type happens to have the same number of characters as the barcode length, which is what the second condition makes sure of

        reconstruction_summary.loc[
            (reconstruction_summary["node_type"].apply(len) == len(self.config.gt_points.gt_df.index[0])) & 
            (~reconstruction_summary["node_type"].isin(self.config.gt_points.gt_df["cell_type"])) & 
            (~reconstruction_summary["node_type"].str.contains("pseudo", na=False)),
            "node_type"
        ] = "unknown_cell"

        reconstruction_summary.loc[
            reconstruction_summary["node_bc"].isin(self.predicted_cell_types.index),
            "type_prediction_score"
        ] = reconstruction_summary["type_prediction_score"].map(self.predicted_cell_types["prediction_score"])
        reconstruction_summary.loc[
            ~reconstruction_summary["node_bc"].isin(self.predicted_cell_types.index),
            "type_prediction_score"
        ] = -1


        for i, col in enumerate(["type_prediction_score", "node_type", "node_bc"]): # just moves the columns to the left
            column_to_move = reconstruction_summary.pop(col)
            reconstruction_summary.insert(1, f"{col}", column_to_move)
            
        self.reconstruction_summary = reconstruction_summary

        self.save_output_file(reconstruction_summary, type = save_type, file_type = "full_reconstruction_summary", enriched=enriched)

    def save_output_file(self, df_to_save="none", type = "unspecified", file_type = "unspecified", enriched=False, mode = "save"):
        '''
        This function is used to save a nd check for multiple types of files, and therefore has quite extensive logic and name generation to find the correct files for eeach occasion
        '''
        save_location = replace_first_folder(self.config.subgraph_location, "Output_files") + f"_{self.reconstruction_dimension}D"
        ensure_directory(save_location)
        if file_type == "quality_df":
            filename = type
            full_save_path = f"{save_location}/{filename}.csv"
        elif file_type =="full_reconstruction_summary":
            filename = f"{type}_{self.name[:-8]}"
            if self.config.modification_type =="dbscan" and self.is_modified_subgraph:
                filename = f"{type}"
            full_save_path = f"{save_location}/{filename}.csv"
        elif self.is_modified_subgraph:

            filename = self.edgelist_location.rsplit("/", 1)[0]
            full_save_path = filename + ".csv"
            if self.config.modification_type =="enriched" and self.is_modified_subgraph:
                location = self.edgelist_location.rsplit("/", 2)[0]
                filename = self.edgelist_location.rsplit("/", 1)[-1]
                save_location = replace_first_folder(location, "Output_files") + f"_{self.reconstruction_dimension}D"
                full_save_path = save_location + "/detailed_edgelist_" + filename
        else:
            filename = f"{type}_subgraph_{self.number}_N={self.n_nodes}"
            full_save_path = f"{save_location}/{filename}.csv"
        print(f"{mode}:",full_save_path)
        # Since the saving logic is the same you would use to chekc if the file exists, these clause make sure it is able to do that
        if mode == "save":
            df_to_save.to_csv(full_save_path, index = False)
        elif mode == "check":
            if os.path.isfile(full_save_path):
                return full_save_path
            else:
                return False

    def generate_detailed_edgefile(self, generate_new_file = False, save_type = "", modified = False):
        file_check = self.save_output_file(type = "detailed_edgelist", mode = "check")
        # If the file exists, do not regenerate it unless specfically told so
        if file_check and not generate_new_file:
            print("Found pre-generated detail edgelist")
            detailed_edgelist = pd.read_csv(file_check)
            self.detailed_edgelist = detailed_edgelist
            return 
        print("Generating detailed edgelist")
        if self.re_filter: # They have slightly different naming conventions
            detailed_edgelist = self.edgelist.copy()
        else:
            detailed_edgelist = self.original_edgelist.copy()

        detailed_edgelist["source_bc"] = detailed_edgelist["source"].map(self.bc_to_idx) # Create barcode columns
        detailed_edgelist["target_bc"] = detailed_edgelist["target"].map(self.bc_to_idx)
        
        detailed_edgelist["source_type"] = detailed_edgelist["source_bc"]
        detailed_edgelist["target_type"] = detailed_edgelist["target_bc"]

        # This for loop replaces the types with the correct ones, based on the barcode
        for col in ["source_type", "target_type"]:

            # Beads and pseudo-cells will have barcodes of different length from the cells, and therefore has to be accounted for in the logic
            detailed_edgelist.loc[
                (detailed_edgelist[col].apply(len) != len(self.config.gt_points.gt_df.index[0])) & 
                (~detailed_edgelist[col].str.contains("pseudo", na=False)),
                col
            ] = "bead"
            
            # If the cell barcode is in the ground truuth df index, it has a ground truth and therefore a cell typ derived from that same df
            detailed_edgelist.loc[
                detailed_edgelist[col].isin(self.config.gt_points.gt_df.index),
                col
            ] = detailed_edgelist[col].map(self.config.gt_points.gt_df["cell_type"])
            
            # all nodes That have the same length as the cells, and are not pseudocells have an unknown cell type
            detailed_edgelist.loc[
                (detailed_edgelist[col].apply(len) == len(self.config.gt_points.gt_df.index[0])) & 
                (~detailed_edgelist[col].str.contains("pseudo", na=False)),
                col
            ] = "unknown_cell"

        # Any edges where the bead happens to be a source, switch so that the bead is always the target
        condition = detailed_edgelist["source_type"] =="bead"
        detailed_edgelist.loc[condition, ["source_type", "target_type"]] = detailed_edgelist.loc[condition, ["target_type", "source_type"]].values
        detailed_edgelist.loc[condition, ["source_bc", "target_bc"]] = detailed_edgelist.loc[condition, ["target_bc", "source_bc"]].values
        detailed_edgelist.loc[condition, ["source", "target"]] = detailed_edgelist.loc[condition, ["target", "source"]].values 

        for i, reconstruction in enumerate(self.all_reconstructions):
            if i==0:
                detailed_edgelist = detailed_edgelist.loc[(detailed_edgelist["source"].isin(reconstruction.index) & detailed_edgelist["target"].isin(reconstruction.index)), :]
            # Align source and target coordinates using reindex to avoid missing key errors
            source_coords = reconstruction.reindex(detailed_edgelist["source"])[["x", "y"]].values
            target_coords = reconstruction.reindex(detailed_edgelist["target"])[["x", "y"]].values

            # Compute Euclidean distance, handling NaNs gracefully
            distances = np.sqrt(np.nansum((source_coords - target_coords) ** 2, axis=1))
            
            # Normalize the distances between 0 and 1 using the max distance
            min_val, max_val = np.nanmin(distances), np.nanmax(distances)
            detailed_edgelist[f"distance_{i+1}"] = (distances - min_val) / (max_val - min_val)

        columns_to_average = [col for col in detailed_edgelist.columns if col.startswith("distance")]
        # Simple statistical metrics on edge lengths
        detailed_edgelist["mean_distance"] = detailed_edgelist[columns_to_average].mean(axis = 1)
        detailed_edgelist["std_distance"] = detailed_edgelist[columns_to_average].std(axis = 1)
        # detailed_edgelist["max_diff_distance"] = columns_to_average.max(axis=1) - columns_to_average.min(axis=1)

        self.detailed_edgelist = detailed_edgelist
        base_edgelist = self.base_edgelist.copy()
        base_edgelist.set_index(["cell_bc_10x", "bead_bc"], inplace=True)

        # Convert edgelist to dictionary for faster lookup speed
        nUMI_dict = base_edgelist["nUMI"].to_dict()

        # Function to retrieve nUMI efficiently
        def get_nUMI(row):
            return nUMI_dict.get((row["source_bc"], row["target_bc"]), None)  # Default to None if not found, as would be the case for i.e pseudocells
        detailed_edgelist["nUMI"] = detailed_edgelist.apply(get_nUMI, axis=1)
        print(detailed_edgelist)

        self.save_output_file(detailed_edgelist, type = "detailed_edgelist")
           
    #endregion
    #region large methods that do a lot of stuff and use a lot of the smaller methods
    def plot_subgraph_enrichment(self, type ="recon"):
        if self.config.filter_analysis_args.network_type !="bi":
            include_gt_edges = False
        else: 
            include_gt_edges = False
        self.max_reconstructions = len(self.all_reconstructions)
        self.find_enriched_subgraph()
        
        print("Starting plotting")
        n_reconstructions = self.config.vizualisation_args.how_many_reconstructions

        if n_reconstructions =="all":
            n_reconstructions = self.max_reconstructions

        if n_reconstructions==2:
            n_reconstructions = 3
        if n_reconstructions == 1 and (self.unenriched_flag or not self.config.subgraph_to_analyse.include_ungated) and self.config.subgraph_to_analyse.gating_threshold !="all":
            single_plotting = True
        else:
            single_plotting = False
        fig, gs, ax_gt_all, ax_gt_subgraph, fontsize, ax_gt_space = self.initialize_figure(type="enrichment", n_reconstructions = n_reconstructions)

        self.all_ground_truth_points.plot_points(ax_gt_all, color_scheme = self.colors, alpha = 0.1, size = 5)
        
        self.plot_ground_truth(ax_gt_all, 1, with_edges=include_gt_edges, size = 8)
     
        # fig.suptitle(f"Subgraph {self.name} from threshold {self.filtering_threshold}")
        ax_gt_all.set_title(f"{self.name} from threshold {self.filtering_threshold}\n{self.n_points_with_gt} points have\nground truth positions", fontsize=fontsize)
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in self.colors.items()]
        ax_gt_all.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),  # Moves it below the plot
            fancybox=True,
            shadow=False,
            ncol=4,  # Number of columns in legend
            fontsize=6,  # Smaller text size
            handleheight=0.8,  # Adjusts spacing
            handlelength=1.5,  # Adjusts patch width
        )
        for recon in range(len(self.all_reconstructions)):
            
            if recon == n_reconstructions:
                break
            if type == "morphed_recon" or type== "morphed_distortion":
                current_reconstruction = self.all_aligned_morphed_reconstructions[recon] 
                self.current_knn = self.knn_all_morphed_reconstructions[recon]
                cpd = self.cpd_all_morphed_reconstructions[recon]
                knn = self.mean_knn_all_morphed_reconstructions[recon]
                all_mean_knn = self.mean_knn_all_morphed_reconstructions
                all_mean_cpd = self.cpd_all_morphed_reconstructions
            else:
                current_reconstruction = self.all_aligned_reconstructions[recon]
                self.current_knn = self.knn_all_reconstructions[recon]
                cpd = self.cpd_all_reconstructions[recon]
                knn = self.mean_knn_all_reconstructions[recon]
                all_mean_knn = self.mean_knn_all_reconstructions
                all_mean_cpd = self.cpd_all_reconstructions
            print(current_reconstruction)
            self.current_knn = self.knn_all_reconstructions[recon]
            self.current_distortion = self.dstrn_all_reconstructions[recon]
            if single_plotting:
                gs_position = gs[0,2]
                title_pos = 1

            else:
                gs_position = gs[recon+1, ax_gt_space]
                title_pos = 0.95
            edges = self.generate_plottable_edges(current_reconstruction, self.original_edgelist, dimension = self.reconstruction_dimension)
            if self.reconstruction_dimension ==3:
                ax_recon_subgraph = fig.add_subplot(gs_position, projection="3d")
                ax_recon_subgraph.set_zticks([])
                ax_recon_subgraph.set_aspect("equal")

                if edges:
                    ax_recon_subgraph.add_collection3d(edges)
            else:
                self.current_dstrn = self.dstrn_all_reconstructions[recon]
                ax_recon_subgraph = fig.add_subplot(gs_position)
                ax_recon_subgraph.set_aspect("equal")
                if self.config.subgraph_to_analyse.include_ungated:
                    ax_recon_subgraph.add_collection(edges)
            if self.config.subgraph_to_analyse.include_ungated:
                if type == "recon" or type=="morphed_recon":
                    self.plot_reconstruction(current_reconstruction, ax_recon_subgraph)
                elif type == "distortion" or type =="morphed_distortion":
                    self.plot_distortion(current_reconstruction, ax_recon_subgraph, subsample = self.config.vizualisation_args.subsample_distortion)
            
                ax_recon_subgraph.set_xticks([])
                ax_recon_subgraph.set_yticks([])
                ax_recon_subgraph.set_title(f"KNN: {knn:.2f} CPD: {cpd:.2f}", fontsize = fontsize, y = title_pos)
                    

            top_cpd_axis = ax_recon_subgraph
            ax_top_mean_cpd_axis = ax_gt_subgraph

        ax_gt_subgraph.set_title(f"Base filter: {self.filtering_threshold}\n KN\u0305N: {np.mean(all_mean_knn):.2f} CP\u0305D: {np.mean(all_mean_cpd):.2f}\n {self.n_all_nodes} nodes \n{self.n_gt_cells}/{self.n_cells} cells with gt", fontsize = fontsize)
        if self.config.subgraph_to_analyse.include_ungated:
            self.plot_ground_truth(ax_gt_subgraph, with_edges=include_gt_edges)
        else:
            ax_gt_enrichedSubgraph = ax_gt_subgraph

        enrichment_positions = ax_gt_space+1

        if self.unenriched_flag == True:
            print(f"Subgraph not enriched")
            
            self.save_plot(fig, type = f"unmodified_{self.config.vizualisation_args.reconstruction_type}", color_scheme=self.config.vizualisation_args.color_scheme)
            return fig

        top_mean_cpd = 0
        top_cpd = 0

        for enrichedSubgraph in self.all_enrichedSubgraphs:
            if self.config.vizualisation_args.color_scheme =="image":
                enrichedSubgraph.factors_for_image_coloring = self.factors_for_image_coloring
                enrichedSubgraph.binary_image = self.binary_image
            if enrichedSubgraph.unenriched_flag:
                print(f"No reconstruction available for subgraph {enrichedSubgraph.number} from filter {enrichedSubgraph.filtering_threshold} {self.config.modification_type} at w={enrichedSubgraph.enrichment_threshold}")
                continue

            enrichedSubgraph.generate_detailed_edgefile(modified=True)
            if self.config.modification_type == "dbscan":
                enrichedSubgraph.create_and_save_full_reconstruction_summary(save_type= f"full_reconstruction_summary_subgraph_{self.number}_N={self.n_nodes}_{self.config.modification_type}_{enrichedSubgraph.full_parameters}", enriched=True)
            else:
                enrichedSubgraph.create_and_save_full_reconstruction_summary(save_type= f"full_reconstruction_summary_subgraph_{self.number}_N={self.n_nodes}_{self.config.modification_type}_{enrichedSubgraph.enrichment_threshold}", enriched=True)
            # enrichedSubgraph.additional_initialization_and_saving_reconstruction_metrics(save_type= f"reconstruction_summary_points_with_gt_subgraph_{self.number}_N={self.n_nodes}_{self.config.modification_type}_{enrichedSubgraph.enrichment_threshold}")

            if enrichedSubgraph.reconstruction_dimension != 2:
                enrichedSubgraph.all_aligned_reconstructions = enrichedSubgraph.all_reconstructions
            if not single_plotting:
                ax_gt_enrichedSubgraph = fig.add_subplot(gs[0, enrichment_positions])
                ax_gt_enrichedSubgraph.set_xticks([])
                ax_gt_enrichedSubgraph.set_yticks([])
                ax_gt_enrichedSubgraph.set_aspect("equal")
            else:
                ax_gt_enrichedSubgraph= ax_gt_subgraph
            

            print(f"{len(enrichedSubgraph.all_aligned_reconstructions)} reconstructions available for subgraph {enrichedSubgraph.number} from filter {enrichedSubgraph.filtering_threshold} {self.config.modification_type} at w={enrichedSubgraph.enrichment_threshold}")
            for recon in range(len(enrichedSubgraph.all_aligned_reconstructions)):
                if recon == n_reconstructions:
                    break

                if type == "morphed_recon" or type== "morphed_distortion":
                    current_reconstruction = enrichedSubgraph.all_aligned_morphed_reconstructions[recon] 
                    enrichedSubgraph.current_knn = enrichedSubgraph.knn_all_morphed_reconstructions[recon]
                    cpd = enrichedSubgraph.cpd_all_morphed_reconstructions[recon]
                    knn = enrichedSubgraph.mean_knn_all_morphed_reconstructions[recon]
                    all_mean_knn = enrichedSubgraph.mean_knn_all_morphed_reconstructions
                    mean_cpd = np.mean(enrichedSubgraph.cpd_all_morphed_reconstructions)
                else:
                    current_reconstruction = enrichedSubgraph.all_aligned_reconstructions[recon]
                    enrichedSubgraph.current_knn = enrichedSubgraph.knn_all_reconstructions[recon]
                    cpd = enrichedSubgraph.cpd_all_reconstructions[recon]
                    knn = enrichedSubgraph.mean_knn_all_reconstructions[recon]
                    all_mean_knn = enrichedSubgraph.mean_knn_all_reconstructions
                    mean_cpd = np.mean(enrichedSubgraph.cpd_all_reconstructions)

                edges = enrichedSubgraph.generate_plottable_edges(current_reconstruction, enrichedSubgraph.edgelist, dimension = enrichedSubgraph.reconstruction_dimension)

                if enrichedSubgraph.reconstruction_dimension ==3:
                    
                    ax_recon_subgraph = fig.add_subplot(gs[recon+1, enrichment_positions], projection="3d")
                    ax_recon_subgraph.set_zticks([])
                    ax_recon_subgraph.set_aspect("equal")
                    if edges:
                        ax_recon_subgraph.add_collection3d(edges)
                else:
                    if not single_plotting:
                        ax_recon_subgraph = fig.add_subplot(gs[recon+1, enrichment_positions])
                        ax_recon_subgraph.set_aspect("equal")
                    ax_recon_subgraph.add_collection(edges)

                ax_recon_subgraph.set_xticks([])
                ax_recon_subgraph.set_yticks([])
                # ax_recon_subgraph = fig.add_subplot(gs[recon+1, enrichment_positions])
                # ax_recon_subgraph.set_xticks([])
                # ax_recon_subgraph.set_yticks([])
                # ax_recon_subgraph.set_box_aspect(1)
                if type == "recon" or type=="morphed_recon":

                    enrichedSubgraph.plot_reconstruction(current_reconstruction, ax_recon_subgraph)
                elif (type == "distortion" or type =="morphed_distortion") and enrichedSubgraph.reconstruction_dimension==2:
                    enrichedSubgraph.plot_distortion(current_reconstruction, ax_recon_subgraph, subsample = self.config.vizualisation_args.subsample_distortion)
                else:
                    enrichedSubgraph.plot_reconstruction(current_reconstruction, ax_recon_subgraph)

                # enrichedSubgraph.plot_reconstruction(current_reconstruction, ax_recon_subgraph)
                # edges = enrichedSubgraph.generate_plottable_edges(current_reconstruction, enrichedSubgraph.edgelist)
                # ax_recon_subgraph.add_collection(edges)
                if top_cpd<cpd:
                    top_cpd=cpd
                    top_cpd_axis = ax_recon_subgraph
                ax_recon_subgraph.set_title(f"KNN: {knn:.2f} CPD: {cpd:.2f}", fontsize = fontsize)

            enrichment_positions +=1
            if include_gt_edges:
                edges = enrichedSubgraph.generate_plottable_edges(enrichedSubgraph.gt_positions, enrichedSubgraph.edgelist)
                ax_gt_enrichedSubgraph.add_collection(edges)
            if mean_cpd>top_mean_cpd:
                top_mean_cpd = mean_cpd
                ax_top_mean_cpd_axis = ax_gt_enrichedSubgraph
            enrichedSubgraph.plot_ground_truth(ax_gt_enrichedSubgraph, with_edges=include_gt_edges)
            ax_gt_enrichedSubgraph.set_title(f"{self.config.modification_type} at: {enrichedSubgraph.enrichment_threshold}\nKN\u0305N: {np.mean(all_mean_knn):.2f} CP\u0305D: {mean_cpd:.2f}\n {enrichedSubgraph.n_all_nodes} nodes\n{enrichedSubgraph.n_gt_cells}/{enrichedSubgraph.n_cells} cells with gt", fontsize = fontsize)
        self.highlight_plot(top_cpd_axis)
        self.highlight_plot(ax_top_mean_cpd_axis)

        self.save_plot(fig, type = f"{self.config.modification_type}-{type}_{self.config.vizualisation_args.how_many_reconstructions}-recon_{self.config.subgraph_to_analyse.gating_threshold}", color_scheme=self.config.vizualisation_args.color_scheme)

    def highlight_plot(self, axis, clr = "blue", width = 2):
        for spine in axis.spines.values():
            spine.set_edgecolor(clr)   # Set the color of the border
            spine.set_linewidth(width)       # Set the width of the border

    def save_plot(self, fig, format=None, type="Unspecified", dpi = 300, color_scheme="cell_type"):
        if format == None:
            format = self.config.vizualisation_args.save_to_image_format

        self.image_location = replace_first_folder(self.config.subgraph_location, "Images") + f"_{self.reconstruction_dimension}D"
        ensure_directory(self.image_location)
        if color_scheme == "diagram":
            filename = f"{type}_subgraph_{self.number}_N={self.n_nodes}"
        else:
            filename = f"colors={color_scheme}_subgraph_{self.number}_N={self.n_nodes}_{type}"
        print(f"{self.image_location}/{filename}.{format}")
        
        fig.savefig(f"{self.image_location}/{filename}.{format}", format = format, dpi = dpi)
        if not self.config.vizualisation_args.show_plots:
            plt.close("all")
    #endregion

class enrichedSubgraph(subgraphToAnalyse):
    '''
    This subclass uses the methods and properties of the bsae subgraph, except the way of finding reconstrutions and other files are slighlty different due to the files generally being one folder deeper
    It also uses some slightly different or additional properties but has no unique methods since the goal is able to be handled as the normal subgraph, with a few extra functionalitites.
    Most importantly, it also shares the properties of the initalising subgraph.
    '''

    def __init__(self, Subgraph, location = None):
        self.__dict__.update(Subgraph.__dict__)
        self.is_modified_subgraph = True
        self.refilter = False
        self.all_unaligned_morphed_reconstructions = None
        if self.config.modification_type == "gated":
            self.re_filter = True

        self.edgelist_location = location
        if self.config.modification_type =="enriched":
            self.enrichment_threshold= int(re.search(r"w=(\d+)", self.edgelist_location).group(1))
            self.reconstruction_location = replace_first_folder(self.edgelist_location, "Subgraph_reconstructions")
            self.reconstruction_location = f"{Path(self.reconstruction_location).parent.parent}_{self.reconstruction_dimension}D/{Path(self.reconstruction_location).parts[-2]}/{Path(self.reconstruction_location).parts[-1]}"
        elif self.config.modification_type =="gated":
            print(self.edgelist_location)
            all_matches = re.findall(r"_gated_(\d+)", self.edgelist_location)
            if len(all_matches)>1:
                n_matches = len(all_matches) 

                self.enrichment_threshold = "+".join(all_matches[:ceil(n_matches/2)])
            else:
                self.enrichment_threshold= str(re.search(r"_gated_(\d+)", self.edgelist_location).group(1))

            self.reconstruction_location = replace_first_folder(self.edgelist_location, "Subgraph_reconstructions")
            self.reconstruction_location = f"{Path(self.reconstruction_location).parent.parent}/{Path(self.reconstruction_location).parts[-2]}/{Path(self.reconstruction_location).parts[-1]}"
        
        elif self.config.modification_type =="dbscan":
            all_matches = re.findall(r"_dbscan_ms=(\d+)", self.edgelist_location)
            _, _, full_parameters = self.edgelist_location.split()[-1].partition("dbscan_")
            self.full_parameters = full_parameters.split("/", 1)[0][:-3]
            if len(all_matches)>1:
                n_matches = len(all_matches) 

                self.enrichment_threshold = "+".join(all_matches[:ceil(n_matches/2)])
            else:
                self.enrichment_threshold= str(re.search(r"_dbscan_ms=(\d+)", self.edgelist_location).group(1))

            self.reconstruction_location = replace_first_folder(self.edgelist_location, "Subgraph_reconstructions")
            self.reconstruction_location = f"{Path(self.reconstruction_location).parent.parent}/{Path(self.reconstruction_location).parts[-2]}/{Path(self.reconstruction_location).parts[-1]}"

        self.edgelist = pd.read_csv(self.edgelist_location)
        self.original_edgelist = self.edgelist.copy()
        if "pseudo=True" in self.edgelist_location:
            pseudo_cells_mapping = pd.read_csv(f"{Path(self.edgelist_location).parent}/pseudo_cells_mapping.csv").set_index("barcode")
            self.pseudo_cells_mapping = pseudo_cells_mapping["Values"].to_dict()
            data = list(self.pseudo_cells_mapping.keys())
            self.bc_to_idx = self.bc_to_idx | self.pseudo_cells_mapping

        self.networkx_graph = nx.Graph()
        for _, row in self.edgelist.iterrows():
            self.networkx_graph.add_edge(row['source'], row['target'])
        if Path(self.reconstruction_location).parent.exists():
            reconstruction_files = [file for file in os.listdir(Path(self.reconstruction_location).parent) if Path(location).name[:-4] in file]
            self.all_reconstructions = [pd.read_csv(f"{Path(self.reconstruction_location).parent}/{recon}").set_index("node_ID") for recon in reconstruction_files]
            self.edgelist = self.edgelist[
                self.edgelist["source"].isin(self.all_reconstructions[0].index) & self.edgelist["target"].isin(self.all_reconstructions[0].index)
            ]
            self.point_ids = self.all_reconstructions[0].index.values

        else:

            self.unenriched_flag = True

def initialize_post_subgraph_analysis(config, initial = False):
    print("Initializing files")
    nUMI_thresholds = config.base_network_args.run_parameters.nUMI_sum_per_bead_thresholds
    n_connections_thresholds = config.base_network_args.run_parameters.n_connected_cells_thresholds
    per_edge_weight_threshold = config.base_network_args.run_parameters.per_edge_nUMI_thresholds
    
    config.subgraph_base_location = f"Subgraph_edgelists/{config.sample_name}/run={config.base_network_args.unfiltered_edge_file[:-4]}_filters=numi{nUMI_thresholds[0]}-{nUMI_thresholds[1]}_nconn{n_connections_thresholds[0]}-{n_connections_thresholds[1]}_w{per_edge_weight_threshold}"
    if initial:
        return config
    
    config.subgraph_location = f"{config.subgraph_base_location}/{config.filter_analysis_args.network_type}-{config.filter_analysis_args.filter}_{config.subgraph_to_analyse.threshold}"
    try:
        if config.subgraph_to_analyse.all_subgraphs:        
            config.all_base_subgraph_files = [f for f in os.listdir(config.subgraph_location) if os.path.isfile(os.path.join(config.subgraph_location, f)) and "unw" in f and ".csv" in f and "enriched" not in f]
        else:
            config.all_base_subgraph_files = [f for f in os.listdir(config.subgraph_location) if os.path.isfile(os.path.join(config.subgraph_location, f)) and "unw" in f and ".csv" in f and int(re.search(r"subgraph_(\d+)", f).group(1)) in config.subgraph_to_analyse.subgraph_number and "enriched" not in f]
    except:
        config.all_base_subgraph_files = []

    config.all_base_subgraph_files = [file for file in config.all_base_subgraph_files if int(re.search(r"N=(\d+)", file).group(1))>=config.subgraph_to_analyse.minimum_subgraph_size]
    if config.all_base_subgraph_files == []:
        print("No subgraphs found, nothing to analyze")
        print("Inspect sugbraph choice criteria")
        print(f"Network type:{config.filter_analysis_args.network_type}")
        print(f"Filter type {config.filter_analysis_args.filter}")
        if not config.filter_analysis_args.analyse_all_thresholds:
            print(f"Thresholds to analyse: {config.filter_analysis_args.thresholds_to_analyse}")
        else:
            print(f"Analyse all thresholds: {config.filter_analysis_args.analyse_all_thresholds}")
        
        if not config.subgraph_to_analyse.all_subgraphs:
            print(f"Analyse all subgraphs: {config.subgraph_to_analyse.all_subgraphs}")
        else:
            print(f"Subgraph to analyse{config.subgraph_to_analyse.subgraph_number}")
        print(f"Subgraph minumum size: {config.subgraph_to_analyse.minimum_subgraph_size}")
    # config.all_subgraph_files = [f for f in files if f"subgraph_{subgraph_number}_" in f]
    return config 

def analyze_subgraph_enrichment(config):
    all_subgraphs = subgraphCollection(config.subgraph_location, config)
    try:
        base_edgelist = pd.read_csv(f"Intermediary_files/{config.sample_name}/{config.base_network_args.unfiltered_edge_file}")
    except:
        base_edgelist = pd.read_csv(f"Intermediary_files/{config.sample_name}/all_cells_synthetic.csv")

    for subgraph_file in config.all_base_subgraph_files:
        print("Reading edgelist")

        subgraph_edgelist = pd.read_csv(f"{config.subgraph_location}/{subgraph_file}")
        print("Generating Subgraph object")
        subgraph = subgraphToAnalyse(config, subgraph_file, subgraph_edgelist)
        subgraph.base_edgelist = base_edgelist
        subgraph.edgelist_location = config.subgraph_location
        subgraph.all_ground_truth_points = config.gt_points
        subgraph.create_and_save_full_reconstruction_summary()
        # subgraph.additional_initialization_and_saving_reconstruction_metrics()
        subgraph.generate_detailed_edgefile(subgraph.config.subgraph_to_analyse.regenerate_detailed_edges)
        subgraph.plot_subgraph_enrichment(type = config.vizualisation_args.reconstruction_type)
        # subgraph.plot_subgraph_enrichment(type = "dstrn")

        all_subgraphs.subgraphs.append(subgraph)
    

    
def initalize_files(config):
    from Utils import Pointcloud
    nUMI_thresholds = config.base_network_args.run_parameters.nUMI_sum_per_bead_thresholds
    n_connections_thresholds = config.base_network_args.run_parameters.n_connected_cells_thresholds
    per_edge_weight_threshold = config.base_network_args.run_parameters.per_edge_nUMI_thresholds
    print("Sample:", config.sample_name)
    config.files_location = f"Intermediary_files/{config.sample_name}/run={config.base_network_args.unfiltered_edge_file[:-4]}_filters=numi{nUMI_thresholds[0]}-{nUMI_thresholds[1]}_nconn{n_connections_thresholds[0]}-{n_connections_thresholds[1]}_w{per_edge_weight_threshold}"
    config.ground_truth_df = pd.read_csv(f"Input_files/{config.base_network_args.ground_truth_file}")
    if config.sample_name == "new_paper_cr":
        ipt="bead_only"
    else:
        ipt ="GT"
    config.gt_points = Pointcloud(config.ground_truth_df, input_type=ipt)
    config.idx_to_bc = pd.read_csv(f"Intermediary_files/{config.sample_name}/barcode_to_index_mapping_all.csv")
    config.raw_edge_file = f"Intermediary_files/{config.sample_name}/{config.base_network_args.unfiltered_edge_file}"
    try:
        config.raw_edges = pd.read_csv(config.raw_edge_file)
    except:
        config.raw_edge_file = f"Intermediary_files/{config.sample_name}/all_cells_synthetic.csv"
    return config

def perform_analysis_actions(config):

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

        analyze_subgraph_enrichment(config)

    plt.show()

if __name__== "__main__":

    from Utils import *
    from Utils import ConfigLoader

    #config_subgraph_analysis_mouse_hippocampus, config_subgraph_analysis_mouse_embryo, config_subgraph_analysis_tonsil, config_analysis, config_subgraph_analysis
    config = ConfigLoader('config_subgraph_analysis_embryo_uni.py')
    perform_analysis_actions(config)
