import os
import pandas as pd
import numpy as np
from Utils import ensure_directory

class Subgraph_to_modify():
    def __init__(self, filename = None, edgelist = None):
        self.name = filename
        self.original_edgelist = edgelist
        

def generate_analysis_file_locations(config, type = "gated"):

    nUMI_thresholds = config.file_localization_args.run_parameters.nUMI_sum_per_bead_thresholds
    n_connections_thresholds = config.file_localization_args.run_parameters.n_connected_cells_thresholds
    per_edge_weight_threshold = config.file_localization_args.run_parameters.per_edge_nUMI_thresholds

    config.run_directory = f"run={config.file_localization_args.unfiltered_edge_file[:-4]}_filters=numi{nUMI_thresholds[0]}-{nUMI_thresholds[1]}_nconn{n_connections_thresholds[0]}-{n_connections_thresholds[1]}_w{per_edge_weight_threshold}"
    if type == "gated" or type == "dbscan":
        config.run_path_base = f"Output_files/{config.sample_name}/{config.run_directory}"
    else:
        config.run_path_base = f"Subgraph_edgelists/{config.sample_name}/{config.run_directory}"

    return config

def load_enriching_source(config):
    source_file_location = f"Intermediary_files/{config.sample_name}/{config.run_directory}"

    if config.simple_subgraph_enriching_args.enriching_sources.use_same_type:
        subgraph_network_type = config.simple_subgraph_enriching_args.subgraph_to_enrich.bi_or_unipartite
        subgraph_filter_type = config.simple_subgraph_enriching_args.subgraph_to_enrich.filter_type
        if config.simple_subgraph_enriching_args.subgraph_to_enrich.bi_or_unipartite =="bi":
            source_name = [f for f in os.listdir(source_file_location) if subgraph_network_type in f][0]
        else:
            source_name = [f for f in os.listdir(source_file_location) if subgraph_network_type in f and subgraph_filter_type in f][0]
        
        source_edgelist = pd.read_csv(f"{source_file_location}/{source_name}")

        return source_edgelist

def generate_enriched_edges(config, subgraph, edge_source):

    all_subgraph_nodes = subgraph.original_edgelist.stack().unique()

    subgraph_creation_threshold = config.simple_subgraph_enriching_args.subgraph_to_enrich.current_threshold
    print(f"Enriching {subgraph.name} from threshold {subgraph_creation_threshold}")
    for enriching_threshold in config.simple_subgraph_enriching_args.enriching_sources.thresholds:
        if enriching_threshold>=subgraph_creation_threshold:
            continue
        else:
            # print(subgraph.original_edgelist)
            extra_edges =edge_source[edge_source["source"].isin(all_subgraph_nodes) & edge_source["target"].isin(all_subgraph_nodes) & (edge_source["weight"] >= enriching_threshold)]
            # extra_edges = extra_edges[extra_edges["weight"]>=enriching_threshold]
            extra_edges.drop("weight", axis = 1).to_csv(f"{subgraph.enrichment_folder}/{subgraph.name[:-4]}_enriched_at_w={enriching_threshold}.csv", index = False)

def enrich_subgraphs_simple(config):
    source_edgelist = load_enriching_source(config)

    subgraph_choice_args = config.simple_subgraph_enriching_args.subgraph_to_enrich
    config.subgraphs_folder_path = f"{config.run_path_base}/{subgraph_choice_args.bi_or_unipartite}-{subgraph_choice_args.filter_type}_{subgraph_choice_args.current_threshold}"
    
    if subgraph_choice_args.enrich_all_subgraphs:
        all_subgraph_files = [f for f in os.listdir(config.subgraphs_folder_path) if "unw" in f and ".csv" in f]
    else:
        all_subgraph_files = [f for f in os.listdir(config.subgraphs_folder_path) if "unw" in f and ".csv" in f and f"subgraph_{config.simple_subgraph_enriching_args.subgraph_to_enrich.subgraph_number}_" in f]
    all_subgraphs = []
    for subgraph in all_subgraph_files:
        subgraph_edgelist = pd.read_csv(f"{config.subgraphs_folder_path}/{subgraph}")

        subgraph_object = Subgraph_to_modify(subgraph, subgraph_edgelist)
        subgraph_object.enrichment_folder = f"{config.subgraphs_folder_path}/{subgraph[:-4]}_enriched"
        print(subgraph_object.enrichment_folder)
        ensure_directory(f"{config.subgraphs_folder_path}/{subgraph[:-4]}_enriched")
        generate_enriched_edges(config, subgraph_object, source_edgelist)
        all_subgraphs.append(subgraph_object)
    config.all_subgraphs = all_subgraphs
    return config

def adapt_config_for_reconstruction_and_reconstruct(config, type = "enriched"):
    from reconstruction_functions import interpret_config_and_reconstruct
    from Utils import blank
    config.reconstruction_args = blank() #just needs to be a random class object
    config.reconstruction_args.reconstruct_all = "all"
    config.reconstruction_args.weighted_reconstruction = False
    config.reconstruction_args.which_subgraphs = "all"
    config.reconstruction_args.manual_reconstruction = False
    config.enriched_flag = True

    config.reconstruction_args.reconstruction_dimension = config.simple_subgraph_enriching_args.reconstruction_dimension
    if type =="gated":

        config.simple_subgraph_enriching_args = config.perform_distance_gated_double_reconstruction.re_reconstruction_args
        config.reconstruction_args.reconstruction_dimension = config.perform_distance_gated_double_reconstruction.re_reconstruction_args.reconstruction_dimension

    elif type =="dbscan":
        config.simple_subgraph_enriching_args = config.perform_dbscan_edge_filtering.re_reconstruction_args
        config.reconstruction_args.reconstruction_dimension = config.perform_dbscan_edge_filtering.re_reconstruction_args.reconstruction_dimension

    config.reconstruction_args.empty_strnd_folders = config.simple_subgraph_enriching_args.empty_strnd_folders
    config.reconstruction_args.reconstructions_per_edgelist = config.simple_subgraph_enriching_args.reconstruct_n_times
    
    # config.edgelist_original_path = 
    counter = 0
    
    for subgraph in config.all_subgraphs:
    # config.subgraph_path_split = 
        split_edge_path = subgraph.enrichment_folder.split("/")
        config.strnd_output_path = f"STRND_structure/{config.sample_name}/{config.run_directory}/{split_edge_path[-2]}/{split_edge_path[-1]}"
        config.run_path = f"{config.run_path_base}/{split_edge_path[-2]}/{split_edge_path[-1]}"
        config.strnd_input_path = f"STRND_structure/data/edge_lists"
        config.strnd_reconstruction_path = f"STRND_structure/data/reconstructed_positions"
        if type != "enriched":
            path_name = split_edge_path[-1]
            s_list = list(path_name)  # Convert string to list
            s_list[-2] = str(config.reconstruction_args.reconstruction_dimension)  # Replace with new digit
            path_name = ''.join(s_list)  # Convert back to string
        else:
            path_name = split_edge_path[-1]
        config.reconstruction_path = f"Subgraph_reconstructions/{config.sample_name}/{config.run_directory}/{split_edge_path[-2]}/{path_name}"

        config.subgraph_path_split = [subgraph.name]

        print("input        ",subgraph.enrichment_folder)
        print("info output  ",config.strnd_output_path)
        print("strnd input  ",config.strnd_input_path)
        print("strnd recon  ",config.strnd_reconstruction_path)
        print("final output ",config.reconstruction_path)
        # print(config.reconstruction_args.manual_reconstruction)
        print(f"Reconstructing {subgraph.name}")
        interpret_config_and_reconstruct(config, input_type=type)
    return config
    pass

def generate_distance_gated_edges(config, subgraph_object, edges, gating_type = "gmm"):
    if gating_type =="gmm":
        from sklearn.mixture import GaussianMixture
        data_base = edges["mean_distance"]
        gmm = GaussianMixture(n_components=2, random_state=42)
        data = data_base.values.reshape(-1, 1)  # GMM requires data in 2D
        gmm.fit(data)

        # Extract fitted parameters
        means = gmm.means_.flatten()  # Means of the Gaussians
        variances = gmm.covariances_.flatten()  # Variances of the Gaussians

        lower_mean_index = np.argmin(means)  # Index of the Gaussian with the lower mean
        mean = means[lower_mean_index]
        std_dev = np.sqrt(variances[lower_mean_index])  # Standard deviation

        # Calculate the 99th percentile
        from scipy.stats import norm
        percentile = norm.ppf(config.perform_distance_gated_double_reconstruction.subgraph_to_enrich.gate_percentile, loc=mean, scale=std_dev)

    gated_edges = edges[edges["mean_distance"]<=0.1].reset_index()
    print(f"{len(edges)-len(gated_edges)}/{len(gated_edges)} edges longer than", f"{percentile:.2f}")
    final_edgelist = gated_edges[["source", "target"]]
    final_edgelist.to_csv(f"{subgraph_object.gated_edges_folder}/{subgraph_object.name}", index = False)
    print(final_edgelist)

def gate_subgraph_edges(config):
    subgraph_choice_args = config.perform_distance_gated_double_reconstruction.subgraph_to_enrich
    dimension = config.perform_distance_gated_double_reconstruction.reconstruction_dimension
    config.subgraphs_folder_path = f"{config.run_path_base}/{subgraph_choice_args.bi_or_unipartite}-{subgraph_choice_args.filter_type}_{subgraph_choice_args.current_threshold}_{dimension}D"
    if subgraph_choice_args.gate_all_subgraphs:
        all_subgraph_files = [f for f in os.listdir(config.subgraphs_folder_path) if "edgelist" in f and ".csv" in f]
    else:
        all_subgraph_files = [f for f in os.listdir(config.subgraphs_folder_path) if "unw" in f and ".csv" in f and f"subgraph_{config.simple_subgraph_enriching_args.subgraph_to_enrich.subgraph_number}_" in f]
    
    if not config.perform_distance_gated_double_reconstruction.subgraph_to_enrich.recursive_gating:
        all_subgraph_files = [f for f in all_subgraph_files if "gated" not in f]

    all_subgraphs = []
    for subgraph in all_subgraph_files:
        subgraph_edgelist = pd.read_csv(f"{config.subgraphs_folder_path}/{subgraph}")
        gate_percentile = config.perform_distance_gated_double_reconstruction.subgraph_to_enrich.gate_percentile
        subgraph_object = Subgraph_to_modify(subgraph, subgraph_edgelist)
        subgraph_object.gated_edges_folder = f"{config.subgraphs_folder_path}/{subgraph[:-4]}_gated_{str(gate_percentile)[2:]}_{dimension}D"
        subgraph_object.enrichment_folder = subgraph_object.gated_edges_folder
        ensure_directory(subgraph_object.gated_edges_folder)
        generate_distance_gated_edges(config,subgraph_object, subgraph_edgelist)
        
        all_subgraphs.append(subgraph_object)
    config.all_subgraphs = all_subgraphs

def generate_dbscan_filtered_edges(config, subgraph_object:Subgraph_to_modify, edges, gating_type = "dbscan", eps_percent = 5, min_samples = 10):
    from sklearn.cluster import DBSCAN  
    from scipy.spatial.distance import pdist
    print(subgraph_object.reconstruction)
    print(subgraph_object.original_edgelist)
    edgelist = subgraph_object.original_edgelist.copy()
    filtered_edgelist = edgelist.copy()
    reconstructions = subgraph_object.reconstruction.copy()
    edgelist_dict = edgelist.set_index(["source_bc", "target_bc"])["nUMI"].to_dict()

    reconstructions.set_index("node_ID", inplace = True)
    reconstruction_to_analyse = 1
    recon_df = reconstructions.loc[:, [ "node_bc", "node_type", f"recon_x_{reconstruction_to_analyse}", f"recon_y_{reconstruction_to_analyse}"]]
    recon_df = recon_df.rename(columns={f"recon_x_{reconstruction_to_analyse}": "x", f"recon_y_{reconstruction_to_analyse}": "y"})
    cells = recon_df[(recon_df["node_type"]!="bead")]
    max_dist = np.max(pdist(cells.loc[:, ["x", "y"]].values))
    start_idx_pseudocells = config.idx_to_bc["index"].max()+1
    pseudo_cells = {}
    for idx, (cell, type, x, y) in cells.iterrows():
        print(f"clustering cell {idx}")
        
        if type =="bead":
            break
        # elif prediction_score==-1 or type!="B_germinal_center":
        #     continue 

        eps_dist = max_dist*eps_percent/100

        cell_edges = edgelist[edgelist["source_bc"] == cell]
        cell_bead_positions = recon_df[recon_df["node_bc"].isin(cell_edges["target_bc"])].copy()


        cell_bead_positions["nUMI"] = cell_bead_positions["node_bc"].map(
                lambda bc: edgelist_dict.get((cell, bc), None)  # Get nUMI using (source_bc, target_bc) tuple
            )
        X = cell_bead_positions[['x', 'y']].values  # Convert to a NumPy array
        dbscan = DBSCAN(eps=eps_dist, min_samples=min_samples)
        df = cell_bead_positions.copy()
        
        df['cluster'] = dbscan.fit_predict(X)


        # Visualize the clusters
        unique_labels = set(df['cluster'])
        clusters = [cluster for cluster in unique_labels if cluster !=-1]

        #these three lines remove all edges to ebads not in a cluster
        beads_not_in_cluster = df[df["cluster"]==-1]
        mask = (filtered_edgelist["source_bc"] == cell) & (filtered_edgelist["target_bc"].isin(beads_not_in_cluster["node_bc"]))
        filtered_edgelist = filtered_edgelist.drop(filtered_edgelist[mask].index)
        
        cluster_counts = df["cluster"].value_counts(ascending = False)
        top_cluster = cluster_counts.idxmax()  # Gets the value with the highest count
        # most_common_count = cluster_counts.max()    # Gets the highest count 
        # print(clusters)
        n_pseudo_node = 1
        if len(clusters)>1:
            # if not config.perform_dbscan_edge_filtering.generate_pseudocells:
            #     beads_not_in_cluster = df[df["cluster"]==-1]

            for i, cluster in enumerate(cluster_counts.index):
                if cluster == top_cluster or cluster == -1:
                    continue

                
                beads_in_extra_cluster = df[df["cluster"]==cluster]
                mask = (filtered_edgelist["source_bc"] == cell) & (filtered_edgelist["target_bc"].isin(beads_in_extra_cluster["node_bc"]))

                if not config.perform_dbscan_edge_filtering.generate_pseudocells:
                    filtered_edgelist = filtered_edgelist.drop(filtered_edgelist[mask].index)
                else:
                    filtered_edgelist.loc[mask, "source_bc"] = f"{idx}_pseudo_node_number_{n_pseudo_node}"
                    filtered_edgelist.loc[mask, "source"] = start_idx_pseudocells
                    filtered_edgelist.loc[mask, "source_type"] = f"pseudo_{type}"
                    pseudo_cells[start_idx_pseudocells] = f"cell_{idx}_pseudo_node_number_{n_pseudo_node}"
                    start_idx_pseudocells+=1
                    n_pseudo_node+=1
    if pseudo_cells:
        df_pseudo_cells = pd.DataFrame.from_dict(pseudo_cells, orient='index', columns=['Values']).reset_index()
        df_pseudo_cells.rename(columns={'index': 'barcode'}, inplace=True)
        df_pseudo_cells.to_csv(f"{subgraph_object.gated_edges_folder}/pseudo_cells_mapping.csv", index=False)
        print(df_pseudo_cells)
        
    print(edgelist)
    print(filtered_edgelist)
    print(subgraph_object.enrichment_folder)
    print(subgraph_object.name)
    filtered_edgelist.loc[:,["source", "target"]].to_csv(f"{subgraph_object.gated_edges_folder}/{subgraph_object.name}", index=False)

def dbscan_filter_subgraph_edges(config):
    subgraph_choice_args = config.perform_dbscan_edge_filtering.subgraph_to_enrich
    dimension = config.perform_dbscan_edge_filtering.reconstruction_dimension
    config.subgraphs_folder_path = f"{config.run_path_base}/{subgraph_choice_args.bi_or_unipartite}-{subgraph_choice_args.filter_type}_{subgraph_choice_args.current_threshold}_{dimension}D"
    if subgraph_choice_args.gate_all_subgraphs:
        all_subgraph_files = [f for f in os.listdir(config.subgraphs_folder_path) if "edgelist" in f and ".csv" in f and "gated" not in f and "dbscan" not in f]
    else:
        all_subgraph_files = [f for f in os.listdir(config.subgraphs_folder_path) if "unw" in f and ".csv" in f and f"subgraph_{config.simple_subgraph_enriching_args.subgraph_to_enrich.subgraph_number}_" in f and "gated" not in f]

    if subgraph_choice_args.gate_all_subgraphs:
        all_subgraph_reconstructions = [f for f in os.listdir(config.subgraphs_folder_path) if "full_reconstruction_summary" in f and ".csv" in f and "gated" not in f and "dbscan" not in f]
    else:
        all_subgraph_reconstructions = [f for f in os.listdir(config.subgraphs_folder_path) if "full_reconstruction_summary" in f and ".csv" in f and f"subgraph_{config.simple_subgraph_enriching_args.subgraph_to_enrich.subgraph_number}_" in f and "gated" not in f]

    all_subgraphs = []
    for subgraph, reconstruction in zip(all_subgraph_files, all_subgraph_reconstructions):
        subgraph_edgelist = pd.read_csv(f"{config.subgraphs_folder_path}/{subgraph}")
        min_samples = config.perform_dbscan_edge_filtering.dbscan_min_sample
        eps = config.perform_dbscan_edge_filtering.dbscan_eps_percentage
        print(subgraph_edgelist)
        subgraph_object = Subgraph_to_modify(subgraph, subgraph_edgelist)
        subgraph_object.gated_edges_folder = f"{config.subgraphs_folder_path}/{subgraph[:-4]}_dbscan_ms={min_samples}_eps={eps}_pseudo={config.perform_dbscan_edge_filtering.generate_pseudocells}_{dimension}D"
        subgraph_object.enrichment_folder = subgraph_object.gated_edges_folder
        subgraph_object.reconstruction = pd.read_csv(f"{config.subgraphs_folder_path}/{reconstruction}")
        ensure_directory(subgraph_object.gated_edges_folder)
        generate_dbscan_filtered_edges(config,subgraph_object, subgraph_edgelist, eps_percent=eps, min_samples = min_samples)
        all_subgraphs.append(subgraph_object)
    config.all_subgraphs = all_subgraphs
    
def perform_simple_subgraph_enriching(config):
    config.idx_to_bc = pd.read_csv(f"Intermediary_files/{config.sample_name}/barcode_to_index_mapping_all.csv")
    for threshold in config.simple_subgraph_enriching_args.subgraph_to_enrich.threshold:
        config.simple_subgraph_enriching_args.subgraph_to_enrich.current_threshold = threshold 
        generate_analysis_file_locations(config, type = "enriching")
        enrich_subgraphs_simple(config)
        if config.simple_subgraph_enriching_args.reconstruct == True:
            adapt_config_for_reconstruction_and_reconstruct(config)

def perform_distance_gated_double_reconstruction(config):
    config.idx_to_bc = pd.read_csv(f"Intermediary_files/{config.sample_name}/barcode_to_index_mapping_all.csv")
    for threshold in config.perform_distance_gated_double_reconstruction.subgraph_to_enrich.threshold:
        for gating_threshold in config.perform_distance_gated_double_reconstruction.subgraph_to_enrich.gate_percentiles:
            config.perform_distance_gated_double_reconstruction.subgraph_to_enrich.gate_percentile = gating_threshold
            config.perform_distance_gated_double_reconstruction.subgraph_to_enrich.current_threshold = threshold 
            generate_analysis_file_locations(config)
            gate_subgraph_edges(config)
            adapt_config_for_reconstruction_and_reconstruct(config, type ="gated")

def perform_dbscan_gated_double_reconstruction(config):
    config.idx_to_bc = pd.read_csv(f"Intermediary_files/{config.sample_name}/barcode_to_index_mapping_all.csv")
    for threshold in config.perform_dbscan_edge_filtering.subgraph_to_enrich.threshold:
        for min_samples in config.perform_dbscan_edge_filtering.dbscan_min_samples:
            for eps in config.perform_dbscan_edge_filtering.dbscan_eps_percentages:
                config.perform_dbscan_edge_filtering.dbscan_min_sample = min_samples
                config.perform_dbscan_edge_filtering.dbscan_eps_percentage = eps
                config.perform_dbscan_edge_filtering.subgraph_to_enrich.current_threshold = threshold 
                generate_analysis_file_locations(config)
                dbscan_filter_subgraph_edges(config)
                adapt_config_for_reconstruction_and_reconstruct(config, type ="dbscan")
if __name__== "__main__":

    from Utils import *
    config = ConfigLoader("config_subgraph_modification.py")

    perform_simple_subgraph_enriching(config)
    # perform_distance_gated_double_reconstruction(config)
    # performSimpleSubgraphEnrichingParallel(config)
    # config.idx_to_bc = pd.read_csv(f"Intermediary_files/{config.sample_name}/barcode_to_index_mapping_all.csv")
    # perform_dbscan_gated_double_reconstruction(config)