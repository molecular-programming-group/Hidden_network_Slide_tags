import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import LineCollection
from matplotlib import cm
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
from typing import List

import seaborn as sns
from matplotlib.ticker import FuncFormatter


class simulationGroup:
    def __init__(self, config, simulation=[]):
        self.all_simulations: List[fullSimulation] = simulation

    def __iter__(self):
        for simulation in self.all_simulations:
            yield simulation
    
    def add_subgraph(self, new_simulation):
        self.all_simulations.append(new_simulation)
    
    def get_parameters_per_simulation(self):
        self.simulation_data = {
            "cpd": [],
            "knn": [],
            "noise": [],
            "cells": [],
            "beads": [],
            "cells_subgraph": [],
            "beads_subgraph": [],
            "time"          : [],
            "D"             : []
        }
        for simulation in self:

            simulation.noise = simulation.diffusion_args.noise_ratio
            print(f"{simulation.config.reconstruction_path}/reconstruction_1.csv")
            if os.path.isfile(f"{simulation.config.reconstruction_path}/reconstruction_1.csv"):
                simulation.cpd = np.mean(simulation.cpd_all_reconstructions)
                simulation.cpd_std = np.std(simulation.cpd_all_reconstructions)
                simulation.knn = np.mean(simulation.mean_knn_all_reconstructions)
                simulation.knn_std = np.std(simulation.mean_knn_all_reconstructions)
                self.simulation_data["cpd"].append(simulation.cpd)
                self.simulation_data["knn"].append(simulation.knn)
            else:
                self.simulation_data["cpd"].append(0)
                self.simulation_data["knn"].append(0)
            self.simulation_data["noise"].append(simulation.diffusion_args.noise_ratio)
            self.simulation_data["cells"].append(simulation.n_cells_for_name)
            self.simulation_data["beads"].append(simulation.n_beads)
            self.simulation_data["cells_subgraph"].append(simulation.cells_in_subgraph)
            self.simulation_data["beads_subgraph"].append(simulation.beads_in_subgraph)
            self.simulation_data["time"].append(simulation.diffusion_args.time)
            self.simulation_data["D"].append(simulation.diffusion_args.D)
        self.parameters_df = pd.DataFrame(self.simulation_data)
        self.parameters_df.index.name = "simulation"

    def plot_simulation_heatmap(self, x_param="cells", y_param="beads", color_param="noise", cmap="viridis", format="png", all_params = []):
        """
        Plots a heatmap where:
        - x-axis = any chosen parameter (e.g., 'cells')
        - y-axis = any chosen parameter (e.g., 'beads')
        - color = any chosen parameter (e.g., 'noise')

        Parameters:
            df (pd.DataFrame): DataFrame containing simulation results.
            x_param (str): Column name to use for X-axis.
            y_param (str): Column name to use for Y-axis.
            color_param (str): Column name to use for color intensity.
            cmap (str): Colormap for the heatmap.
        """

        # Pivot the DataFrame to structure it for heatmap plotting
        extra_params = [col for col in self.parameters_df.columns if col not in [x_param, y_param, color_param] and col in all_params]
        print(self.parameters_df)
        grouped = self.parameters_df.groupby(extra_params) if extra_params else [(None, self.parameters_df)]
        
        for values, group_df in grouped:
            print(group_df)
            pivot_table = group_df.pivot(index=y_param, columns=x_param, values=color_param)

            # Sort axes in decreasing order (if needed)
            if y_param == "noise":
                ascending_y = True
            else:
                ascending_y = False
            ascending_x = False
            pivot_table = pivot_table.sort_index(ascending=ascending_y)  # Sort Y-axis (beads)
            pivot_table = pivot_table.sort_index(axis=1, ascending=ascending_x)  # Sort X-axis (cells)

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap=cmap, linewidths=0.5, cbar_kws={'label': color_param}, vmin = 0, vmax = 1)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            # Function to format tick labels (round to 3 decimals)
            def round_ticks(x, _):
                return f"{x:.3f}"

            # Apply tick formatting
            # ax.xaxis.set_major_formatter(FuncFormatter(round_ticks))
            # ax.yaxis.set_major_formatter(FuncFormatter(round_ticks))

            # Labels & Title
            plt.xlabel(x_param)
            plt.ylabel(y_param)
            title = f"{color_param} Heatmap: {y_param} vs {x_param}"
            file_name = f"Simulation/heatmap_{x_param}_{y_param}_{color_param}"
            if extra_params:
                title += " | " + ", ".join(f"{p}={v}" for p, v in zip(extra_params, values))
                file_name += "_" + "_".join(f"{p}_{v}" for p, v in zip(extra_params, values))

            plt.title(title)

            plt.savefig(f"{file_name}.{format}", format = format)

class fullSimulation:
    def __init__(self, config, rand_seed = None):
        self.config = config
        self.diffusion_args = self.config.diffusion
        self.rand_seed = rand_seed
        

        self.space_args = self.config.space
        self.shape = self.space_args.shape
        self.dimension = self.space_args.dimension
        self.x_span = self.space_args.x_span
        self.y_span = self.space_args.y_span
        self.z_span = self.space_args.z_span
        self.point_args = self.config.points
        self.n_cells = self.point_args.n_cells
        self.n_cells_for_name = self.config.final_cells

        self.n_beads = self.point_args.n_beads
        
        if self.dimension ==3:
            if self.config.points.bead_mode == "lasagna":
                self.full_run_parameters_str = f"{self.dimension}D_{self.shape}_{self.x_span}x{self.y_span}x{self.z_span}_cells={self.n_cells_for_name}_beads={self.n_beads}_mode={self.config.points.bead_mode}{self.config.points.layers}_diff_D={self.diffusion_args.D}_t={self.diffusion_args.time}_noise={self.diffusion_args.noise_ratio}_edge_gen={self.diffusion_args.edge_gen_mode}_pwrexp={self.diffusion_args.powerlaw_exp}_cell_cutting={self.config.points.generate_n_cells_by_cutting}"
            elif self.config.space.shape in ["cylinder"]:
                self.full_run_parameters_str = f"{self.dimension}D_{self.shape}_{self.x_span}x{self.z_span}_cells={self.n_cells_for_name}_beads={self.n_beads}_mode={self.config.points.bead_mode}_diff_D={self.diffusion_args.D}_t={self.diffusion_args.time}_noise={self.diffusion_args.noise_ratio}_edge_gen={self.diffusion_args.edge_gen_mode}_pwrexp={self.diffusion_args.powerlaw_exp}_cell_cutting={self.config.points.generate_n_cells_by_cutting}"
            else:
                self.full_run_parameters_str = f"{self.dimension}D_{self.shape}_{self.x_span}x{self.y_span}x{self.z_span}_cells={self.n_cells_for_name}_beads={self.n_beads}_mode={self.config.points.bead_mode}_diff_D={self.diffusion_args.D}_t={self.diffusion_args.time}_noise={self.diffusion_args.noise_ratio}_edge_gen={self.diffusion_args.edge_gen_mode}_pwrexp={self.diffusion_args.powerlaw_exp}_cell_cutting={self.config.points.generate_n_cells_by_cutting}"
        elif self.dimension ==2:
            if self.config.points.bead_mode == "lasagna":
                self.full_run_parameters_str = f"{self.dimension}D_{self.shape}_{self.x_span}x{self.y_span}_cells={self.n_cells_for_name}_beads={self.n_beads}_mode={self.config.points.bead_mode}{self.config.points.layers}_diff_D={self.diffusion_args.D}_t={self.diffusion_args.time}_noise={self.diffusion_args.noise_ratio}_edge_gen={self.diffusion_args.edge_gen_mode}_pwrexp={self.diffusion_args.powerlaw_exp}_cell_cutting={self.config.points.generate_n_cells_by_cutting}"
            elif self.config.space.shape in ["cylinder"]:
                self.full_run_parameters_str = f"{self.dimension}D_{self.shape}_{self.x_span}_cells={self.n_cells_for_name}_beads={self.n_beads}_mode={self.config.points.bead_mode}_diff_D={self.diffusion_args.D}_t={self.diffusion_args.time}_noise={self.diffusion_args.noise_ratio}_edge_gen={self.diffusion_args.edge_gen_mode}_pwrexp={self.diffusion_args.powerlaw_exp}_cell_cutting={self.config.points.generate_n_cells_by_cutting}"
            else:
                self.full_run_parameters_str = f"{self.dimension}D_{self.shape}_{self.x_span}x{self.y_span}_cells={self.n_cells_for_name}_beads={self.n_beads}_mode={self.config.points.bead_mode}_diff_D={self.diffusion_args.D}_t={self.diffusion_args.time}_noise={self.diffusion_args.noise_ratio}_edge_gen={self.diffusion_args.edge_gen_mode}_pwrexp={self.diffusion_args.powerlaw_exp}_cell_cutting={self.config.points.generate_n_cells_by_cutting}"

        if self.config.force_points_regeneration or not os.path.exists(f"Simulation/{self.full_run_parameters_str}/node_positions.csv"):
            if self.config.base_on_sample:
                pass
            else:
                self.generate_points(point_type = "cells")
            self.generate_points(point_type = "beads", type =self.config.points.bead_mode)
            
            self.cells_df = pd.DataFrame(self.cells)
            
            
            self.beads_df = pd.DataFrame(self.beads)
            # if self.dimension == 3:
            #     self.cells_df.columns = ["x", "y", "z"]
            #     self.beads_df.columns = ["x", "y", "z"]
            #     extra_cells_df = pd.DataFrame(self.extra_cells.T, columns = ["x", "y", "z"])
            # else:
            #     self.cells_df.columns = ["x", "y"]
            #     self.beads_df.columns = ["x", "y"]
            #     extra_cells_df = pd.DataFrame(self.extra_cells.T, columns = ["x", "y"])
            # extra_cells_df["type"] = "boundary_cell"
            self.beads_df["type"] = "bead"
            self.cells_df["type"] = "cell"
            self.beads_df.index = self.beads_df.index + self.n_cells
            # self.cells_for_bead_generation_df = pd.concat([self.cells_df, extra_cells_df]).reset_index()
            # if self.dimension == 3:
            #     self.cells_for_bead_generation = self.cells_for_bead_generation_df[["x", "y" , "z"]].copy().values
            # else:
            #     self.cells_for_bead_generation = self.cells_for_bead_generation_df[["x", "y"]].copy().values

            all_nodes = pd.concat([self.cells_df, self.beads_df], ignore_index=True)
            self.nodes = all_nodes.reset_index().rename(columns={'index': 'id'})
            self.generate_edges(sampling_type=self.diffusion_args.edge_gen_mode)
            self.save_network()
        else:
            self.nodes = pd.read_csv(f"Simulation/{self.full_run_parameters_str}/node_positions.csv")
            self.cells_df  = self.nodes.loc[self.nodes["type"]=="cell", :].set_index("id")

            self.beads_df  = self.nodes.loc[self.nodes["type"]=="bead", :].set_index("id")
            if self.dimension == 3:
                self.beads_df.columns = ["x", "y", "z", "type"]
                self.cells_df.columns = ["x", "y", "z", "type"]
                self.cells = self.cells_df[["x", "y", "z"]].values
                self.beads = self.beads_df[["x", "y", "z"]].values
            else:
                self.cells = self.cells_df[["0", "1"]].values
                self.beads = self.beads_df[["0", "1"]].values
            save_location = f"Simulation/{self.full_run_parameters_str}"
            
            self.edges_df = pd.read_csv(f"{save_location}/edgelist_simulation.csv")
            self.edges_df_nu_dupes = pd.read_csv( f"{save_location}/edgelist_simulation_no_dupes.csv")
            self.final_edges = self.edges_df.values
            source_counts = self.edges_df_nu_dupes["source"].value_counts()
            valid_sources = source_counts[source_counts > 1].index
            filtered_edges = self.edges_df_nu_dupes[self.edges_df_nu_dupes["source"].isin(valid_sources)]

            # Print result (optional)
            # print(filtered_edges, "huh")
            self.edges_no_unidegree_beads = filtered_edges
            self.edges_no_unidegree_beads_array = filtered_edges.values

    def generate_points(self, point_type = "cells", type = None):
        if point_type == "cells":
            if self.rand_seed:
                np.random.seed(self.rand_seed)
            n_points = self.n_cells
        elif point_type == "beads":
            if self.rand_seed:
                np.random.seed(self.rand_seed+1)
            n_points = self.n_beads
        else:
            n_points = 1

        if self.shape == "rectangle":
            x = np.random.uniform(-self.x_span/2, self.x_span/2, n_points)
            y = np.random.uniform(-self.x_span/2, self.x_span/2, n_points)
        elif self.shape in ["dome", "cylinder"]:
            radius = self.x_span / 2  # Assuming x_span defines the diameter of the circle

            angles = np.random.uniform(0, 2 * np.pi, n_points)
            radii = radius * np.sqrt(np.random.uniform(0, 1, n_points))

            x = radii * np.cos(angles)
            y = radii * np.sin(angles)

            if point_type =="cells":
                buffer_factor = (radius+500)/radius

                extended_radius = radius * buffer_factor
                extra_n = int(n_points * (buffer_factor**2 - 1))
                angles_extra = np.random.uniform(0, 2 * np.pi, extra_n)
                radii_extra = extended_radius * np.sqrt(np.random.uniform(0, 1, extra_n))
                # x_extra = radii_extra * np.cos(angles_extra)
                # y_extra = radii_extra * np.sin(angles_extra)

                # outside_mask = radii_extra > radius  # Keep only points outside the original shape

        if self.dimension ==3 and point_type !="beads":
            if self.shape in ["rectangle", "cylinder"]:
                z = np.random.uniform(0, self.z_span, n_points)
                # if point_type =="cells":
                #     buffer_factor = (self.z_span+500)/self.z_span
                #     z_extra = np.random.uniform(0, self.z_span*buffer_factor, extra_n)
                #     outside_mask = (radii_extra > radius) | (z_extra > self.z_span)
                #     x_extra = x_extra[outside_mask]
                #     y_extra = y_extra[outside_mask]
                #     z_extra = z_extra[outside_mask]  # Keep only external points in z-dimension
                    # fig = plt.figure(figsize=(8, 8))
                    # ax = fig.add_subplot(111, projection='3d')

                    # # Scatter plot for original points (blue)
                    # ax.scatter(x, y, z, s=25, color="blue", alpha=0.5, label="Original Region")

                    # # Scatter plot for extended points (red)
                    # ax.scatter(x_extra, y_extra, z_extra, s=25, color="red", alpha=0.5, label="Extended Region")

                    # # Labels and aspect ratio
                    # ax.set_xlabel("X")
                    # ax.set_ylabel("Y")
                    # ax.set_zlabel("Z")
                    # ax.set_title("Original Region (Blue) vs Extended Region (Red)")
                    # ax.legend()
            elif self.shape in ["dome"]: # not completed
                z = np.random.uniform(0, self.z_span, n_points)

            points = np.vstack((x, y, z)).T   # Shape (n_points, 3)
        elif self.dimension == 3 and point_type =="beads":
            if type == "surface":
                z = np.zeros(n_points)
            elif type == "cloud":
                if self.shape in ["rectangle", "cylinder"]:
                    z = np.random.uniform(0, self.z_span, n_points)
                elif self.shape in ["dome"]: # not completed
                    z = np.random.uniform(0, self.z_span, n_points)
            elif type =="surface2":
                z = np.random.choice([0,self.z_span], n_points)
            elif type =="lasagna":
                z = np.random.choice(np.linspace(0, self.z_span, self.point_args.layers), n_points)
            else:
                z = np.zeros(n_points)
            points = np.vstack((x, y, z)).T  # Shape (n_points, 3)
        else:
            points = np.vstack((x, y)).T  # Shape (n_points, 2)
        if point_type == "cells":
            self.cells = points
            # if self.dimension == 3:
            #     self.extra_cells = np.vstack((x_extra, y_extra, z_extra))
            # else:
            #     self.extra_cells = np.vstack((x_extra, y_extra))
        elif point_type == "beads":
            self.beads = points


    def generate_edges(self, sampling_type = "powerlaw"):
        print("generating edges")
        print(self.full_run_parameters_str)
        D = self.diffusion_args.D
        t = self.diffusion_args.time
        r = np.linspace(0, np.max([self.x_span, self.y_span, self.z_span]), 500)  # Distance range from 0 to 10 (adjust as needed)

        if self.dimension == 2:
            pdf = (1 / (4 * np.pi * D * t)) * np.exp(-r**2 / (4 * D * t))
            label = "2D Fick's Law (Gaussian concentration)"
        elif self.dimension == 3:
            pdf = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(-r**2 / (4 * D * t))
            label = "3D Fick's Law (Gaussian concentration)"
        else:
            raise ValueError("Dimension must be 2 or 3")
        from scipy.ndimage import gaussian_filter1d

        # Compute numerical derivative (first-order difference)
        pdf_smooth = gaussian_filter1d(pdf, sigma=3)  # Smooth to reduce noise
        derivative = np.gradient(pdf_smooth, r)  # Compute derivative
        
        # Find where the slope is close to zero
        if self.dimension ==2:
            threshold = 1e-15  # Adjust based on scale
        else:
            threshold = 1e-15
        
        flat_region_index =0
        while flat_region_index ==0:
            flat_region_index = np.where(np.abs(derivative) < threshold)[0][0]
            threshold /=10
        flat_distance = r[flat_region_index]
            
        if self.rand_seed:
            np.random.seed(self.rand_seed)
        from scipy.stats import zipf
        if sampling_type == "powerlaw":
            
            a = self.diffusion_args.powerlaw_exp  # Power-law exponent (>1)
            max_value = 1000

            x = np.arange(1, max_value + 1)
            pdf = zipf.pmf(x, a)
            valid_indices = np.where(self.cells_for_bead_generation_df["type"] == "cell")[0]
            all_edges = []  # To accumulate all edges
            for bead, id in zip(self.beads, self.beads_df.index):
                while True:
                    n_probes = zipf.rvs(a) -1
                    if n_probes <= max_value:
                        break
                distances = np.linalg.norm(self.cells - bead, axis=1)
                if self.dimension == 2:
                    weights = (1 / (4 * np.pi * D * t)) * np.exp(-distances**2 / (4 * D * t))
                    w_end = (1 / (4 * np.pi * D * t)) * np.exp(-flat_distance**2 / (4 * D * t))
                elif self.dimension == 3:
                    weights = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(-distances**2 / (4 * D * t))
                    w_end = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(-flat_distance**2 / (4 * D * t))

                weights[distances > flat_distance] = w_end
                weights /= weights.sum()
                sampled_indices = []
                for _ in range(n_probes):
                    if np.random.rand() < self.diffusion_args.noise_ratio/100:
                        # Random noise edge
                        sampled_idx = np.random.randint(len(distances))
                    else:
                        # Diffusion-weighted edge
                        sampled_idx = np.random.choice(len(distances), p=weights)

                    # if self.cells_for_bead_generation_df.at[sampled_idx, "type"] =="cell":
                    sampled_indices.append(sampled_idx) 
                real_cell_indices = [idx for idx in sampled_indices if idx in valid_indices]
                # if len(real_cell_indices) != len(sampled_indices):
                #     print(sampled_indices, real_cell_indices)
                edges = np.column_stack((np.full(len(real_cell_indices), id), real_cell_indices))
                all_edges.append(edges)  # Save edges for this bead
        elif sampling_type =="pdf":
            
            valid_indices = np.where(self.cells_df["type"] == "cell")[0]
            all_edges = []  # To accumulate all edges

            for bead, id in zip(self.beads, self.beads_df.index):
                # Compute distances
                distances = np.linalg.norm(self.cells - bead, axis=1) 
                # distances = np.array([0])
                # Compute diffusion-based probabilities
                if self.dimension == 2:
                    weights = (1 / (4 * np.pi * D * t)) * np.exp(-distances**2 / (4 * D * t))
                    w_end = (1 / (4 * np.pi * D * t)) * np.exp(-flat_distance**2 / (4 * D * t))
                    w0 = (1 / (4 * np.pi * D * t)) * np.exp(0**2 / (4 * D * t))
                elif self.dimension == 3:
                    weights = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(-distances**2 / (4 * D * t))
                    w_end = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(-flat_distance**2 / (4 * D * t))
                    w0 = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(0**2 / (4 * D * t))
                # weights[distances > flat_distance] = w_end
                # print(weights)

                # weights /= weights.sum()
                weights *= 0.01/w0
                # print(weights)
                # quit()
                a = self.diffusion_args.powerlaw_exp
                # fig, ax = plt.subplots()
                
                
                # ax.hist(weights, bins = 5000)
                # ax.set_xscale("log")
                # fig, ax = plt.subplots()
                
                # ax.scatter(distances, weights)
                # ax.set_yscale("log")
                # plt.show()
                n_probes=1e6
                while n_probes >1e5:
                    n_probes = zipf.rvs(a)

                # **New Step**: Select each index independently based on weight probabilities
                sampled_indices = np.where(np.random.rand(len(distances)) < 0)[0]
                for _ in range(n_probes):
                    chosen_distances = np.where(np.random.rand(len(distances)) < weights)[0]
                    sampled_indices = np.append(sampled_indices, chosen_distances)
                # Apply noise condition (select some random indices)
                n_noisy = int(len(sampled_indices) * (self.diffusion_args.noise_ratio / 100))
                n_noisy = min(n_noisy, len(distances))  # Limit n_noisy to max possible indices

                # if n_noisy > 0 and len(sampled_indices) > 0:
                #     # Choose which "true" indices will be replaced

                #     # Replace selected true indices with noise indices
                #     replace_indices = np.random.choice(len(sampled_indices), size=n_noisy, replace=False)

                #     # Generate noise indices (random choices from full index range)
                #     noise_indices = np.random.choice(len(distances), size=n_noisy, replace=False)
                #     # Replace the values at the chosen indices with noise indices
                #     sampled_indices[replace_indices] = noise_indices

                # Filter for valid cell types
                real_cell_indices = sampled_indices[np.isin(sampled_indices, valid_indices)]

                # Store edges
                edges = np.column_stack((np.full(len(real_cell_indices), id), real_cell_indices))
                all_edges.append(edges)  # Save edges for this bea

        # Combine all edges into one array

        self.final_edges = np.vstack(all_edges)
        if self.diffusion_args.noise_ratio > 0:
            total_edges = len(self.final_edges)
            n_noisy = int(total_edges * (self.diffusion_args.noise_ratio / 100))

            # Randomly choose which rows (edges) to replace with noise
            noise_rows = np.random.choice(total_edges, size=n_noisy, replace=False)

            # Replace target cell indices with random (valid) indices
            noise_targets = np.random.choice(valid_indices, size=n_noisy, replace=True)

            self.final_edges[noise_rows, 1] = noise_targets  # Replace target column (index 1)
        self.edges_df = pd.DataFrame(self.final_edges, columns = ["source", "target"])
        if config.generate_cells_by_cutting:
            self.cut_cells(n_cells_final=self.config.final_cells)

        self.edges_df_nu_dupes = self.edges_df.drop_duplicates()

        source_counts = self.edges_df_nu_dupes["source"].value_counts()
        valid_sources = source_counts[source_counts > 1].index
        filtered_edges = self.edges_df_nu_dupes[self.edges_df_nu_dupes["source"].isin(valid_sources)]

        # Print result (optional)
        # print(filtered_edges, "huh")
        self.edges_no_unidegree_beads = filtered_edges
        self.edges_no_unidegree_beads_array = filtered_edges.values

    def cut_cells(self, n_cells_final = 10000):
        n_cells_start = len(self.edges_df["target"].unique())
        cell_in_edges = self.edges_df["target"].unique()
        if len(cell_in_edges)>n_cells_final:
            cells_to_keep = np.random.choice(cell_in_edges, size=n_cells_final, replace=False)
        else:
            cells_to_keep = cell_in_edges
        filtered_edges = self.edges_df[self.edges_df["target"].isin(cells_to_keep)].copy()
        self.edges_df = filtered_edges
        self.final_edges = self.edges_df.values.copy()

        filtered_cell_df = self.cells_df.loc[self.cells_df.index.isin(cells_to_keep)].copy()
        self.cells_df = filtered_cell_df
        filtered_nodes = self.nodes.loc[self.nodes.index.isin(cells_to_keep) | (self.nodes["type"] == "bead")].copy()
        self.nodes = filtered_nodes

    def plot_diffusion_pdf(self):
        D = self.diffusion_args.D
        t = self.diffusion_args.time
        dimension = self.dimension
        r = np.linspace(0, np.max([self.x_span, self.y_span, self.z_span]), 500)  # Distance range from 0 to 10 (adjust as needed)

        if dimension == 2:
            pdf = (1 / (4 * np.pi * D * t)) * np.exp(-r**2 / (4 * D * t))
            label = "2D Fick's Law (Gaussian concentration)"
        elif dimension == 3:
            pdf = (1 / ((4 * np.pi * D * t) ** 1.5)) * np.exp(-r**2 / (4 * D * t))
            label = "3D Fick's Law (Gaussian concentration)"
        else:
            raise ValueError("Dimension must be 2 or 3")
        from scipy.ndimage import gaussian_filter1d

        # Compute numerical derivative (first-order difference)
        pdf_smooth = gaussian_filter1d(pdf, sigma=3)  # Smooth to reduce noise
        derivative = np.gradient(pdf_smooth, r)  # Compute derivative

        # Find where the slope is close to zero
        if self.dimension ==2:
            threshold = 1e-10  # Adjust based on scale
        else:
            threshold = 1e-15
        flat_region_index = np.where(np.abs(derivative) < threshold)[0][0]
        corner_distance = r[flat_region_index]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(r, pdf, label=f"{dimension}D Fick's Law (Gaussian concentration)\nD={D}, t={t}")
        plt.axvline(corner_distance, color='red', linestyle="--", label=f"Bottom Corner ≈ {corner_distance:.2f}")
        plt.xlabel("Distance (r)")
        plt.ylabel("Probability Density")
        plt.title("Diffusion Probability Density Function")
        plt.legend()
        # plt.grid(True)

    def plot_simple_points(self, points, s = 10, c = "k"):
        if points.shape[1] == 3:
            # 3D case
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, s=s)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        else:
            # 2D case
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.scatter(points[:, 0], points[:, 1], c=c, s=s)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")
    
    def plot_cells_and_beads(self, ax = None, color_scheme = None, edges = True, beads = True):
        if self.cells.shape[1] == 3:
            # 3D case
            if ax == None:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111, projection='3d')

            if color_scheme == "x":
                colors = self.cells[:, 0]  # Color by X
                cmap = cm.viridis
            elif color_scheme == "y":
                colors = self.cells[:, 1]  # Color by Y
                cmap = cm.viridis
            elif color_scheme == "z":
                colors = self.cells[:, 2]  # Color by Z
                cmap = cm.viridis
            elif color_scheme == "r":  # Radial distance
                colors = np.sqrt(self.cells[:, 0]**2 + self.cells[:, 1]**2 + self.cells[:, 2]**2)
                cmap = cm.viridis
            else:
                colors = "k"  # Default to black if no valid color_scheme

            # Normalize colors if using a colormap
            if isinstance(colors, np.ndarray):
                norm = plt.Normalize(colors.min(), colors.max())
                colors = cmap(norm(colors))  # Apply colormap
            ax.scatter(self.cells[:, 0], self.cells[:, 1], self.cells[:, 2], c=colors, s=10, linewidths=1)
            if beads:
                ax.scatter(self.beads[:, 0], self.beads[:, 1], self.beads[:, 2], c="gray", s=3)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_aspect("equal")  # Equal aspect ratio
            edge_lines = []
            if edges and beads:
                for bead_id, cell_idx in self.final_edges:
                    if "x" in self.beads_df.columns:
                        bead_pos = self.beads_df.loc[int(bead_id), ["x", "y", "z"]].values
                        cell_pos = self.cells_df.loc[int(cell_idx), ["x", "y", "z"]].values
                    else:
                        bead_pos = self.beads_df.loc[int(bead_id), [0,1,2]].values
                        cell_pos = self.cells_df.loc[int(cell_idx), [0,1,2]].values
                    edge_lines.append([bead_pos, cell_pos])
                edge_collection = Line3DCollection(edge_lines, colors='k', linewidths=0.5, alpha=0.1)
                ax.add_collection3d(edge_collection)
        else:
            
            # 2D case
            
            if ax == None:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            if color_scheme == "x":
                colors = self.cells[:, 0]  # Color by X
                cmap = cm.viridis
            elif color_scheme == "y":
                colors = self.cells[:, 1]  # Color by Y
                cmap = cm.viridis
            elif color_scheme == "r":  # Radial distance
                colors = np.sqrt(self.cells[:, 0]**2 + self.cells[:, 1]**2)
                cmap = cm.viridis
            else:
                colors = "k"  # Default to black if no valid color_scheme
            ax.scatter(self.cells[:, 0], self.cells[:, 1], c=colors, s=10, linewidths=1)
            if beads:
                ax.scatter(self.beads[:, 0], self.beads[:, 1], c="lightgray", s=2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_aspect("equal")
            if edges and beads:
                edge_lines = []
                for bead_id, cell_idx in self.final_edges:
                    if all(col in self.beads_df.columns for col in ["0", "1"]):
                        bead_pos = self.beads_df.loc[int(bead_id), ["0","1"]].values
                        cell_pos = self.cells_df.loc[int(cell_idx), ["0","1"]].values
                    else:
                        bead_pos = self.beads_df.loc[int(bead_id), [0,1]].values
                        cell_pos = self.cells_df.loc[int(cell_idx), [0,1]].values

                    edge_lines.append([bead_pos[:2], cell_pos[:2]])  # Only x, y in 2D

                edge_collection = LineCollection(edge_lines, colors='gray', linewidths=0.5, alpha=0.5)
                ax.add_collection(edge_collection)

    def save_network(self):

        ensure_directory("Simulation")
        print(self.full_run_parameters_str)
        save_location = f"Simulation/{self.full_run_parameters_str}"
        ensure_directory(save_location)

        positions_location = f"{save_location}/node_positions.csv"
        self.nodes.to_csv(positions_location, index = False)
        self.edges_df.to_csv( f"{save_location}/edgelist_simulation.csv", index = False)
        self.edges_df_nu_dupes.to_csv( f"{save_location}/edgelist_simulation_no_dupes.csv", index = False)
        self.edges_no_unidegree_beads.to_csv( f"{save_location}/edgelist_simulation_no_dupes_nu_unidegree_beads.csv", index = False)
    def initalize_reconstruction(self, print_outputs = True):
        from network_spatial_coherence.structure_and_args import GraphArgs
        
        config = self.config
        config.strnd_output_path = f"STRND_structure/Simulation/{self.full_run_parameters_str}/STRND_output.csv"
        config.strnd_input_path = f"STRND_structure/data/edge_lists"
        config.strnd_reconstruction_path = f"STRND_structure/data/reconstructed_positions"
        config.reconstruction_path = f"Simulation/{self.full_run_parameters_str}"
        config.edgelist_original_path = f"Simulation/{self.full_run_parameters_str}/edgelist_simulation_no_dupes_nu_unidegree_beads.csv"
        if print_outputs:
            print("input        ",config.edgelist_original_path)
            print("info output  ",config.strnd_output_path)
            print("strnd input  ",config.strnd_input_path)
            print("strnd recon  ",config.strnd_reconstruction_path)
            print("final output ",config.reconstruction_path)
            print(config.strnd_input_path)
        # shutil.copy(config.edgelist_original_path, config.strnd_input_path) # Has to be after reconstruction initialization to have the STRND structure present
        config.working_directory = os.getcwd() #This is for finding the correct output path for the STRND structure
        print()
        args = GraphArgs(data_dir=f"{config.working_directory}/STRND_structure")
        # args = GraphArgs(data_dir=f"C:/Users/simon.kolmodin/Desktop/Slide_Tag/Publication_code/STRND_structure")
        shutil.copy(config.edgelist_original_path, config.strnd_input_path)

        # define conditions for the run
        args.spatial_coherence_validation['gram_matrix'] = False
        args.spatial_coherence_validation['network_dimension'] = False
        args.spatial_coherence_validation['spatial_constant'] = False

        args.show_plots = False
        args.colorfile = "dna_cool2.png"
        args.plot_original_image = False
        args.reconstruct = True
        args.reconstruction_mode = 'STRND'
        args.plot_reconstructed_image = False

        args.proximity_mode = "experimental"
        args.edge_list_title = "edgelist_simulation_no_dupes_nu_unidegree_beads.csv"
        args.dim = self.config.reconstruction.recon_dimension
        args.verbose = False
        self.recon_args = args
        return args

    def perform_reconstruction(self, n_recons = 1):
        import network_spatial_coherence.nsc_pipeline as nsc
        
        args = self.recon_args
        graph, args = nsc.load_and_initialize_graph(args=args)
        for i in range(n_recons):
            single_graph_args, output_df = nsc.run_pipeline(graph, args)
            # ensure_directory(config.strnd_output_path)
            # output_df.to_csv(f"{config.strnd_output_path}/{config.subgraph_path_split[-1][:-4]}_recon_info.csv", index= False)
            output_df.set_index('Property', inplace=True)

            from Utils import ensure_directory
            if output_df.at['num_points', 'Value'] == len(self.nodes.index):
                reconstruction_file = [f for f in os.listdir(config.strnd_reconstruction_path) if f"N={output_df.at['num_points', 'Value']}" in f][0]
            else:
                reconstruction_file = [f for f in os.listdir(config.strnd_reconstruction_path) if f"N={output_df.at['num_points', 'Value']}" in f and "old_index" in f][0]
            ensure_directory(config.reconstruction_path)
            reconstruction_n = i+1
            shutil.copy(f"{config.strnd_reconstruction_path}/{reconstruction_file}", f"{config.reconstruction_path}/reconstruction_{reconstruction_n}.csv")
            # while True:
            #     if not os.path.isfile(f"{config.reconstruction_path}/reconstruction_{reconstruction_n}.csv"):
            #         shutil.copy(f"{config.strnd_reconstruction_path}/{reconstruction_file}", f"{config.reconstruction_path}/reconstruction_{reconstruction_n}.csv")
            #         break
            #     else:
            #         reconstruction_n += 1

        for filename in os.listdir(config.strnd_input_path):
            file_path = os.path.join(config.strnd_input_path, filename)
            if os.path.isfile(file_path):  # Check if it's a file
                os.remove(file_path)
        for filename in os.listdir(config.strnd_reconstruction_path):
            file_path = os.path.join(config.strnd_reconstruction_path, filename)
            if os.path.isfile(file_path):  # Check if it's a file
                os.remove(file_path)

    def load_reconstructions(self, check = False):
        print("Searching reconstructions")
        reconstruction_files = [file for file in os.listdir(f"Simulation/{self.full_run_parameters_str}") if "reconstruction" in file]
        if reconstruction_files:
            self.reconstructed=True
        else:
            print("No reconstructions found")
            self.reconstructed = False
            self.beads_in_subgraph = 0
            self.cells_in_subgraph = 0
            return
        self.all_reconstructions = []
        for i, reconstruction in enumerate(reconstruction_files):
            print(f"Reconstruction {i+1} found")
            reconstructed_positions = pd.read_csv(f"Simulation/{self.full_run_parameters_str}/{reconstruction}")
            if self.dimension == 3:
                if len(reconstructed_positions.columns) ==3:
                    reconstructed_positions["z"] = np.zeros(len(reconstructed_positions["x"]))
                    reconstructed_positions.columns = [f"r{i+1}_x", f"r{i+1}_y", "id", f"r{i+1}_z"]

                else:
                    reconstructed_positions.columns = [f"r{i+1}_x", f"r{i+1}_y", f"r{i+1}_z", "id"]
            elif self.dimension == 2:
                reconstructed_positions.columns = [f"r{i+1}_x", f"r{i+1}_y", f"id"]
            reconstructed_positions.set_index("id", inplace=True)
            self.all_reconstructions.append(reconstructed_positions)
             
            self.nodes =  self.nodes.merge(reconstructed_positions, on="id", how="left")
            
        if reconstruction_files:
            self.reconstructed = True
            self.beads_in_subgraph = self.nodes[(self.nodes["type"] == "bead") & self.nodes["r1_x"].notna()].shape[0]
            self.cells_in_subgraph = self.nodes[(self.nodes["type"] == "cell") & self.nodes["r1_x"].notna()].shape[0]

            if not os.path.isfile(f"Simulation/{self.full_run_parameters_str}/quality_summary.csv"):
                self.calculate_cpd()
                self.calculate_knn()
                
                quality_dict = {
                    "knn_means": self.mean_knn_all_reconstructions,
                    "knn_std": self.std_knn_all_reconstructions,
                    "knn_medians": self.median_knn_all_reconstructions,
                    "cpds" :self.cpd_all_reconstructions

                }
                quality_df = pd.DataFrame(quality_dict)
                quality_df.to_csv(f"Simulation/{self.full_run_parameters_str}/quality_summary.csv", index = False)
            else:
                quality_df = pd.read_csv(f"Simulation/{self.full_run_parameters_str}/quality_summary.csv")
                self.mean_knn_all_reconstructions = quality_df["knn_means"].values
                self.std_knn_all_reconstructions= quality_df["knn_std"].values
                self.median_knn_all_reconstructions= quality_df["knn_medians"].values
                self.cpd_all_reconstructions= quality_df["cpds"].values
            self.align_reconstructions()


    def align_reconstructions(self):
        all_aligned_reconstructions = []
        gt_positions_all = self.nodes
        for i, reconstruction in enumerate(self.all_reconstructions):

            # Since the reconstruction might not only have points with a ground truth such as beads, we have to extract only the points which have a corresponding ground truth point
            matching_indexes = self.nodes.index.intersection(reconstruction.index)
            recon_with_gt = reconstruction.loc[matching_indexes]
            if self.dimension ==3:
                if "x" in gt_positions_all.columns:
                    gt_with_recon = gt_positions_all[[f"x", f"y", f"z"]].loc[matching_indexes]
                elif "0" in gt_positions_all.columns:
                    gt_with_recon = gt_positions_all[["0","1","2"]].loc[matching_indexes]
                else:
                    gt_with_recon = gt_positions_all[[0,1,2]].loc[matching_indexes]
            else:
                if all(col in gt_positions_all.columns for col in ["0", "1"]):
                    gt_with_recon = gt_positions_all[["0", "1"]].loc[matching_indexes]
                else:
                    gt_with_recon = gt_positions_all[[0, 1]].loc[matching_indexes]
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
        self.all_aligned_reconstructions = all_aligned_reconstructions

    def plot_reconstructions(self, color_scheme = None, edges = True, beads = True):
        full_nodes = self.nodes.copy().set_index("id")
        fig = plt.figure(figsize=(12,6))
        if self.dimension ==3:
            first_ax = fig.add_subplot(1, 2, 1, projection='3d')  # First subplot, 3D
        else:
            first_ax = fig.add_subplot(1,2, 1)
        first_ax.set_aspect ("equal")

        
        self.plot_cells_and_beads(ax = first_ax, color_scheme= color_scheme, edges = False, beads = beads)

        n_subgraphs = len(self.all_reconstructions)
        max_columns = 6 
        figsize = 6

        if n_subgraphs < max_columns:
            max_columns = n_subgraphs

        n_rows = (n_subgraphs + max_columns - 1) // max_columns  # Calculate the required rows
        
        # fig = plt.figure(figsize=(max_columns * figsize, n_rows * figsize))

        # First plot (only one on the first row, aligned to the left)
        for i, recon in enumerate(self.all_aligned_reconstructions):
            
            reconstructed_cells = self.cells_in_subgraph
            if color_scheme == "x":
                colors = self.cells_df.values[:, 0]  # Color by X
                cmap = cm.viridis
            elif color_scheme == "y":
                colors = self.cells_df.values[:, 1]  # Color by Y
                cmap = cm.viridis
            elif self.dimension ==3 and color_scheme == "z":
                colors = self.cells_df.values[:, 2]  # Color by Z
                cmap = cm.viridis
            elif color_scheme == "r":  # Radial distance
                if self.dimension ==3:
                    colors = np.sqrt(self.cells_df.values[:, 0]**2 + self.cells_df.values[:, 1]**2 + self.cells_df.values[:, 2]**2)
                else:
                    colors = np.sqrt(self.cells_df.values[:, 0]**2 + self.cells_df.values[:, 1]**2)
                cmap = cm.viridis
            else:
                colors = "k"  # Default to black if no valid color_scheme
            colors = np.array(colors, dtype=float)
            colors_df = pd.DataFrame({"id":self.cells_df.index, "color" : colors}).set_index("id")
            # Normalize colors if using a colormap
            if isinstance(colors, np.ndarray):
                norm = plt.Normalize(colors.min(), colors.max())
                colors = cmap(norm(colors))  # Apply colormap


            if self.dimension ==3 and reconstructed_cells[f"r{i+1}_z"].sum() != 0:
                # ax = fig.add_subplot(n_rows, max_columns, i+1, projection='3d')
                ax = fig.add_subplot(n_rows, 2, 2, projection='3d')

                recon_cells = full_nodes.loc[(full_nodes["type"]=="cell") & (full_nodes.index.isin(recon.index))]
                ax.scatter(recon_cells[f"r{i+1}_x"], recon_cells[f"r{i+1}_y"], recon_cells[f"r{i+1}_z"], c = colors_df[recon_cells.index], s = 10)
                recon_beads = full_nodes.loc[(full_nodes["type"]=="bead") & (full_nodes.index.isin(recon.index))]
                if beads:
                    ax.scatter(recon_beads[f"r{i+1}_x"], recon_beads[f"r{i+1}_y"], recon_beads[f"r{i+1}_z"], c = "k", s = 3, alpha = 0.2)

                if edges:
                    edge_lines = []
                    for bead_id, cell_idx in self.edges_no_unidegree_beads_array:
                        if bead_id in recon_beads.index and cell_idx in recon_cells.index:
                            bead_pos = recon_beads.loc[int(bead_id), [f"r{i+1}_x", f"r{i+1}_y", f"r{i+1}_z"]].values
                            cell_pos = recon_cells.loc[int(cell_idx), [f"r{i+1}_x", f"r{i+1}_y", f"r{i+1}_z"]].values
                            edge_lines.append([bead_pos, cell_pos])
                    edge_collection = Line3DCollection(edge_lines, colors='k', linewidths=0.5, alpha=0.1)
                    ax.add_collection3d(edge_collection)
            else:
                # ax = fig.add_subplot(n_rows, max_columns, i+1)
                ax = fig.add_subplot(n_rows, 2, 2)

                recon_cells = full_nodes.loc[(full_nodes["type"]=="cell") & (full_nodes.index.isin(recon.index))]
                if len(colors)!=1:
                    plot_colors = colors_df.loc[recon_cells.index].values
                else:
                    plot_colors = "k"
                ax.scatter(recon_cells[f"r{i+1}_x"], recon_cells[f"r{i+1}_y"], c = plot_colors, s = 15)
                recon_beads = full_nodes.loc[(full_nodes["type"]=="bead") & (full_nodes.index.isin(recon.index))]
                if beads:
                    ax.scatter(recon_beads[f"r{i+1}_x"], recon_beads[f"r{i+1}_y"], c = "gray", s = 3, alpha = 0.2)
                if edges:
                    edge_lines = []
                    for bead_id, cell_idx in self.edges_no_unidegree_beads_array:
                        if bead_id in recon_beads.index and cell_idx in recon_cells.index:
                            bead_pos = recon_beads.loc[int(bead_id), [f"r{i+1}_x", f"r{i+1}_y"]].values
                            cell_pos = recon_cells.loc[int(cell_idx),[f"r{i+1}_x", f"r{i+1}_y"]].values
                            edge_lines.append([bead_pos, cell_pos])
                    edge_collection = LineCollection(edge_lines, colors='k', linewidths=0.5, alpha=0.1)
                    ax.add_collection(edge_collection)
            ax.set_aspect("equal")
            cpd = self.cpd_all_reconstructions[i]
            knn = self.mean_knn_all_reconstructions[i]
            ax.set_title(f"CPD:{cpd:.3f} KNN:{knn:.3f}, {len(recon_cells)} cell {len(recon_beads)} beads")
        format = "png"
        plt.suptitle(self.full_run_parameters_str)
        fig.savefig(f"Simulation/{self.full_run_parameters_str}/recon_positions_color_{color_scheme}.{format}", format = format)

    def plot_degree_distributions(self):
        fig, axes = plt.subplots(2, 2, figsize = (12, 12))
        if self.edges_df.empty:
            return
        # colors = ["b", "g", "r", "m", "y", "k", "c"]
        self.calculate_degrees()
        ax_cells = axes[0, 0]
        ax_beads_degrees = axes[0,1]
        ax_beads_umis = axes[1,1]
        ax_cells.scatter(self.cell_degree_dist.index, self.cell_degree_dist.values, s = 4)
        # ax_cells.axvline(self.mean_cell_degrees, label = f"mean degree")
        ax_beads_degrees.scatter(self.bead_degree_dist.index, self.bead_degree_dist.values, s = 4)
        self.bead_umi_dist
        # ax_beads.axvline(self.mean_bead_degrees, label = f"mean degree")
        ax_beads_umis.scatter(self.bead_umi_dist.index, self.bead_umi_dist.values, s = 4)

        ax_cells.set_ylabel("Count")
        ax_cells.set_xlabel("Degree")
        ax_cells.set_xscale("log")
        ax_cells.set_yscale("log")
        # ax_cells.legend(fontsize = 4)
        ax_cells.set_title(f"Cells N = {self.cell_degree_dist.values.sum()}")
        ax_cells.set_box_aspect(1)

        ax_beads_degrees.set_ylabel("Count")
        ax_beads_degrees.set_xlabel("Degree")
        ax_beads_degrees.set_xscale("log")
        ax_beads_degrees.set_yscale("log")
        # ax_beads_degrees.legend(fontsize = 4)
        ax_beads_degrees.set_title(f"Beads N = {self.bead_degree_dist.values.sum()}")
        ax_beads_degrees.set_box_aspect(1)

        ax_beads_umis.set_ylabel("Count")
        ax_beads_umis.set_xlabel("UMIs")
        ax_beads_umis.set_xscale("log")
        ax_beads_umis.set_yscale("log")
        # ax_beads_umis.legend(fontsize = 4)
        # ax_beads_degrees.set_title(f"UMIs")
        ax_beads_umis.set_box_aspect(1)
        format = "png"
        fig.savefig(f"Simulation/{self.full_run_parameters_str}/degree_distributions.{format}", format = format)

    def calculate_degrees(self):
        # Count degrees for each source
        n_umis_beads = self.edges_df["source"].value_counts()
        self.bead_umi_dist = n_umis_beads.value_counts()
        self.bead_degrees =  self.edges_df_nu_dupes["source"].value_counts()
        self.bead_degree_dist = self.bead_degrees.value_counts()
        # Count degrees for each target
        self.cell_degrees = self.edges_df_nu_dupes["target"].value_counts()
        self.cell_degree_dist = self.cell_degrees.value_counts()

    def calculate_cpd(self):
        # If desired, you can set an upper limit to how many points to calculate if more speed is desired at the cost of a bit of accuracy 
        # note that none of the samples have more than 10 000 ground truth points (cells) and therefore the default settings does nothing since it only counts the ground truth positions
        gt_positions_all = self.nodes.copy().set_index("id")
        self.cpd_all_reconstructions, self.reconstructed_pairwise_distances = [], []
        for i, reconstruction in enumerate(self.all_reconstructions):
            recon_cells = self.nodes.loc[(self.nodes["type"]=="cell") & (self.nodes["id"].isin(reconstruction.index))]
            gt_df = recon_cells.copy().set_index("id")
            matching_indexes = gt_df.index.intersection(reconstruction.index)
            recon_with_gt = reconstruction.loc[matching_indexes]
            print(gt_positions_all)
            if self.dimension ==3:
                recon_with_gt_for_tree = recon_with_gt[[f"r{int(i)+1}_x", f"r{int(i)+1}_y", f"r{int(i)+1}_z"]]
                if "x" in gt_positions_all.columns:
                    gt_with_recon = gt_positions_all[[f"x", f"y", f"z"]].loc[matching_indexes]
                elif "0" in gt_positions_all.columns:
                    gt_with_recon = gt_positions_all[["0","1","2"]].loc[matching_indexes]
                else:
                    gt_with_recon = gt_positions_all[[0,1,2]].loc[matching_indexes]
            else:
                recon_with_gt_for_tree = recon_with_gt[[f"r{int(i)+1}_x", f"r{int(i)+1}_y"]]
                if all(col in gt_positions_all.columns for col in ["0", "1"]):
                    gt_with_recon = gt_positions_all[["0", "1"]].loc[matching_indexes]
                else:
                    gt_with_recon = gt_positions_all[[0, 1]].loc[matching_indexes]

            try:

                original_distances = pdist(gt_with_recon) #Calculate all pairwise distances for the ground truth positions
                self.gt_pairwise_distances = original_distances
                reconstructed_distances = pdist(recon_with_gt_for_tree)#Calculate all pairwise distances for the all reconstructed positions with a ground truth
                self.reconstructed_pairwise_distances.append(reconstructed_distances)
                if len(reconstructed_distances)<3: # if there are only two points, correlation will always be 1 otherwise
                    correlation = 0
                else:
                    correlation, _ = pearsonr(original_distances, reconstructed_distances) 
            except:
                print("RAM error")
                correlation = 0
            r_squared = correlation**2

            self.cpd_all_reconstructions.append(r_squared)

    def calculate_knn(self, k = 15):
        from scipy.spatial import KDTree
        # If desired, you can set an upper limit to how many points to calculate if more speed is desired at the cost of a bit of accuracy 
        # note that none of the samples have more than 10 000 ground truth points (cells) and therefore the default settings does nothing since it only counts the ground truth positions
        gt_positions_all = self.nodes.copy().set_index("id")
        self.knn_all_reconstructions, self.mean_knn_all_reconstructions, self.median_knn_all_reconstructions, self.std_knn_all_reconstructions = [], [], [], []
        for i, reconstruction in enumerate(self.all_reconstructions):
            recon_cells = self.nodes.loc[(self.nodes["type"]=="cell") & (self.nodes["id"].isin(reconstruction.index))]
            gt_df = recon_cells.copy().set_index("id")
            matching_indexes = gt_df.index.intersection(reconstruction.index)
            recon_with_gt = reconstruction.loc[matching_indexes]
            print(gt_positions_all)
            if self.dimension ==3:
                recon_with_gt_for_tree = recon_with_gt[[f"r{int(i)+1}_x", f"r{int(i)+1}_y", f"r{int(i)+1}_z"]]
                if "x" in gt_positions_all.columns:
                    gt_with_recon = gt_positions_all[[f"x", f"y", f"z"]].loc[matching_indexes]
                elif "0" in gt_positions_all.columns:
                    gt_with_recon = gt_positions_all[["0","1","2"]].loc[matching_indexes]
                else:
                    gt_with_recon = gt_positions_all[[0,1,2]].loc[matching_indexes]
            else:
                recon_with_gt_for_tree = recon_with_gt[[f"r{int(i)+1}_x", f"r{int(i)+1}_y"]]
                if all(col in gt_positions_all.columns for col in ["0", "1"]):
                    gt_with_recon = gt_positions_all[["0", "1"]].loc[matching_indexes]
                else:
                    print(gt_positions_all)
                    gt_with_recon = gt_positions_all[[0, 1]].loc[matching_indexes]

            original_tree = KDTree(gt_with_recon)  # construct the KD tree, 
            original_neighbors = original_tree.query(gt_with_recon, k + 1)[1][:, 1:] # k+1 since it counts itself, otherwise we would only get k-1 neighbours

            reconstructed_tree = KDTree(recon_with_gt_for_tree)
            reconstructed_neighbors = reconstructed_tree.query(recon_with_gt_for_tree, k + 1)[1][:, 1:]
            knn_per_point = []
            for original, reconstructed in zip(original_neighbors, reconstructed_neighbors):  # each row will be a unique cell, so we loop over them to acquire the ratio of shared points betweek the two             
                n = len(original)
                knn_per_point.append(len(set(original).intersection(set(reconstructed[:n]))) / n)
            
            self.knn_all_reconstructions.append(knn_per_point)
            self.mean_knn_all_reconstructions.append(np.mean(knn_per_point))
            self.median_knn_all_reconstructions.append(np.median(knn_per_point))    
            self.std_knn_all_reconstructions.append(np.std(knn_per_point))

    def plot_degree_vs_z(self):

        nUMIs_cells = self.edges_df["target"].value_counts()
        z_cells = self.nodes.loc[nUMIs_cells.index, "z"]

        fig, ax = plt.subplots(1,1, figsize = (6,6))
        ax.scatter(z_cells, nUMIs_cells.values)
        ax.set_xscale("log")
if __name__== "__main__":
    from Utils import *
    config_base = ConfigLoader("config_simulation", type="simulation")
    quit()
    all_simulations = simulationGroup(config_base.copy())
    total = len(config_base.diffusion.D)*len(config_base.diffusion.all_powerlaw_exp)*len(config_base.diffusion.times)*len(config_base.diffusion.noise_ratios)*len(config_base.points.all_n_beads)*len(config_base.points.all_n_cells)

    L = 1
    for diff_const in config_base.diffusion.D:
        for pwr in config_base.diffusion.all_powerlaw_exp:
            for time in config_base.diffusion.times:
                for noise in config_base.diffusion.noise_ratios:
                    for nbeads in config_base.points.all_n_beads:
                        for ncells in config_base.points.all_n_cells:
                            config = config_base.copy()
                            percent = (L / total) * 100  # Calculate percentage
                            bar = "#" * (L) + "-" * ((total) - (L))  # Create bar
                            print(f"\r[{bar}] {percent:.1f}% ({L}/{total})", end="", flush=True)  # Print on the same line
                            L +=1
                            config.diffusion.powerlaw_exp = pwr
                            if config_base.points.generate_n_cells_by_cutting:
                                if config_base.points.starting_cells < ncells:
                                    config.points.n_cells = ncells
                                    config.final_cells = ncells
                                else:
                                    config.points.n_cells = config_base.points.starting_cells
                                    config.final_cells = ncells
                            else:
                                config.points.n_cells = ncells
                                config.final_cells = ncells
                            config.diffusion.D = diff_const
                            config.generate_cells_by_cutting = config_base.points.generate_n_cells_by_cutting
                            config.points.n_beads = nbeads
                            config.diffusion.time = time
                            config.diffusion.noise_ratio = noise
                            simulation = fullSimulation(config, rand_seed = 42)
                            # simulation.plot_diffusion_pdf()
                            # plt.show()
                            # simulation.plot_simple_points(simulation.cells, c = "lightgray")
                            # simulation.plot_simple_points(simulation.beads, s = 5, c = "k")
                            # os.path.isfile(f"{simulation.config.reconstruction_path}/reconstruction_1.csv")
                            
                            # simulation.plot_cells_and_beads(edges = True)
                            # # plt.show()
                            # simulation.plot_degree_distributions()
                            # plt.show()
                            simulation.initalize_reconstruction(print_outputs = False)
                            simulation.load_reconstructions(check = True)
                            
                            if simulation.config.reconstruct =="dev" and not simulation.reconstructed:
                                simulation.config.reconstruct = True

                            if simulation.config.reconstruct == True:
                                
                                for i in range(simulation.config.reconstruction.n_reconstructions):
                                    
                                    if os.path.isfile(f"{simulation.config.reconstruction_path}/reconstruction_{i+1}.csv"):
                                        continue
                                    simulation.perform_reconstruction(n_recons = simulation.config.reconstruction.n_reconstructions)
                                    simulation.load_reconstructions()
                                    plt.rcdefaults()  # Resets all rcParams to default
                                    
                            
                            color_list = ["x"]
                            for color_scheme in color_list:
                                if simulation.reconstructed:
                                    beads = False
                                    simulation.plot_reconstructions(color_scheme=color_scheme, edges = False, beads = beads)
                                    plt.close("all")  # Closes all active figures
                                    # plt.show()
                                else:
                                    print("No reconstructions")
                            # simulation.plot_degree_vs_z()
                            all_simulations.add_subgraph(simulation)

    bar = "#" * ((total))  # Create bar
    print(f"\r[{bar}] {100.0:.1f}%")  # Print on the same line
    # 
    all_simulations.get_parameters_per_simulation()
    all_params = ["cells", "noise", "beads", "time", "D"]
    all_simulations.plot_simulation_heatmap(x_param="cells", y_param="noise", color_param="cpd", all_params=all_params, format = "png", cmap = "rocket")
    all_simulations.plot_simulation_heatmap(x_param="cells", y_param="noise", color_param="knn", all_params=all_params, format = "png", cmap = "rocket")
    plt.show()   