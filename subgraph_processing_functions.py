import pandas as pd
import networkx as nx
import os
from Utils import generate_directory_names, ensure_subgraph_filtering_directory
'''
These functions convert the BC edgelists to index ones instead (BC --> number for each unique barcode) 
'''
def convert_edgelist_format(config):
    barcode_mapping = pd.read_csv(f'{config.sample_path}/barcode_to_index_mapping_all.csv') #This index is created in the "initial_processing_functions" step since it should be done on the unfiltered set of barcodes
    barcode_dict = dict(zip(barcode_mapping['barcode'], barcode_mapping['index'])) #make it a dict so its faster to lookup

    if os.path.isfile(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_bipartite.csv"): #check if it exists for this sample and these conditions already
        print(f"Found processed bipartite edgefile:\n   {config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_bipartite.csv\n")

    else:
        filename_post_filter_edgelist_bipartite = f"{config.sample_path}/{config.filtering_directory}/{config.sample_name}_edge_list_filtered_by_per_edge_weight_{config.filtering_args.per_edge_nUMI_thresholds}.csv"
        edge_list_bipartite = pd.read_csv(filename_post_filter_edgelist_bipartite)

        edge_list_bipartite['cell_bc_10x'] = edge_list_bipartite['cell_bc_10x'].map(barcode_dict) #these two lines maps replaces all barcodes with their corresponding number
        edge_list_bipartite['bead_bc'] = edge_list_bipartite['bead_bc'].map(barcode_dict)
        edge_list_bipartite.columns = ["source", "target", "weight"] #name it according to common network notation, in this case source is always cells and targets beads
        edge_list_bipartite.to_csv(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_bipartite.csv", index = False)

    #The only difference in the following two similar step is that all barcodes are cell barcodes and has two different weight metrics from the two different adjancency matrix types
    #It uses networkX to convert from adjacency list to edgelist
    if os.path.isfile(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_beads.csv"):
        print(f"Found processed beadweight unipartite edgefile:\n   {config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_beads.csv\n")

    else: 
        if os.path.isfile(f"{config.sample_path}/{config.filtering_directory}/cell-cell_umiweight_matrix.csv"):
            filename_post_filter_adjacency_unipartite = f"{config.sample_path}/{config.filtering_directory}/cell-cell_beadweight_matrix.csv"
            adjacency_unipartite = pd.read_csv(filename_post_filter_adjacency_unipartite, index_col = "Unnamed: 0")
            G = nx.from_pandas_adjacency(adjacency_unipartite)

            # Convert the graph to an edgelist
            edge_list_unipartite = nx.to_pandas_edgelist(G)

            edge_list_unipartite['source'] = edge_list_unipartite['source'].map(barcode_dict)
            edge_list_unipartite['target'] = edge_list_unipartite['target'].map(barcode_dict)
            edge_list_unipartite.to_csv(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_beads.csv", index = False)
        else:
            print("No Unipartite bead-weight matrix found")

    if os.path.isfile(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_umis.csv"):
        print(f"Found processed umiweight unipartite edgefile:\n   {config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_umis.csv\n")

    else:
        if os.path.isfile(f"{config.sample_path}/{config.filtering_directory}/cell-cell_umiweight_matrix.csv"):
            filename_post_filter_adjacency_unipartite = f"{config.sample_path}/{config.filtering_directory}/cell-cell_umiweight_matrix.csv"
            adjacency_unipartite = pd.read_csv(filename_post_filter_adjacency_unipartite, index_col = "Unnamed: 0")
            G = nx.from_pandas_adjacency(adjacency_unipartite)

            # Convert the graph to an edgelist
            edge_list_unipartite = nx.to_pandas_edgelist(G)

            edge_list_unipartite['source'] = edge_list_unipartite['source'].map(barcode_dict)
            edge_list_unipartite['target'] = edge_list_unipartite['target'].map(barcode_dict)
            edge_list_unipartite.to_csv(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_umis.csv", index = False)
        else:
            print("No Unipartite UMI-weight matrix found")

def load_edgelist(config, type):    #This functions just loads the correct file depending on the exact context

    if type == "bi":
        edge_list = pd.read_csv(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_bipartite.csv")
        return edge_list

    elif type == "uni_beads":
        edge_list_beads = pd.read_csv(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_beads.csv")
        return edge_list_beads
    
    elif type == "uni_umis":
        edge_list_umis = pd.read_csv(f"{config.sample_path}/{config.filtering_directory}/bc_to_idx_edgelist_unipartite_umis.csv")
        return edge_list_umis

def find_subgraphs(config, edge_list):
    '''
    This functions finds subgraphs as networkX objects and puts them in a list
    '''
    G = nx.from_pandas_edgelist(edge_list, source='source', target='target', edge_attr="weight")

    # Find all connected components (subgraphs)
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) >= config.subgraph_processing_args_filtering.minimum_subgraph_size] 

    return subgraphs


def filter_bipartite(config):
    edge_list = load_edgelist(config, "bi")
    for threshold in config.subgraph_processing_args_filtering.numi_filters_bi:
        nbeads_filtered_edges = edge_list[edge_list["weight"]>=threshold]
        subgraphs = find_subgraphs(config, nbeads_filtered_edges)
        # print(len(subgraphs))
        if len(subgraphs)>0:
            config = ensure_subgraph_filtering_directory(config, "bi-umis", threshold)
            save_path = config.run_path
        else:
            continue
        n_subgraph = 0
        for subgraph in subgraphs:
            n_subgraph +=1
            subgraph_edgelist = nx.to_pandas_edgelist(subgraph)
            subgraph_edgelist.to_csv(f"{save_path}/subgraph_{n_subgraph}_N={subgraph.number_of_nodes()}.csv", index = False)
            subgraph_edgelist.drop("weight", axis = 1).to_csv(f"{save_path}/subgraph_{n_subgraph}_N={subgraph.number_of_nodes()}_unw.csv", index = False)
            
def filter_unipartite_beads(config, threshold):
    edge_list_beads = load_edgelist(config, "uni_beads")

    nbeads_filtered_edges = edge_list_beads[edge_list_beads["weight"]>=threshold]
    subgraphs = find_subgraphs(config, nbeads_filtered_edges)
    if len(subgraphs)>0:
        config = ensure_subgraph_filtering_directory(config, "uni-beads", threshold)
        save_path = config.run_path
    else:
        return
    n_subgraph = 0

    for subgraph in subgraphs:
        n_subgraph +=1
        print(subgraph)
        subgraph_edgelist = nx.to_pandas_edgelist(subgraph)
        subgraph_edgelist.to_csv(f"{save_path}/subgraph_{n_subgraph}_N={subgraph.number_of_nodes()}.csv", index = False)
        subgraph_edgelist.drop("weight", axis = 1).to_csv(f"{save_path}/subgraph_{n_subgraph}_N={subgraph.number_of_nodes()}_unw.csv", index = False)
    
def filter_unipartite_uMIs(config, threshold):
    edge_list_umis = load_edgelist(config, "uni_umis")

    nbeads_filtered_edges = edge_list_umis[edge_list_umis["weight"]>=threshold]
    subgraphs = find_subgraphs(config, nbeads_filtered_edges)
    if len(subgraphs)>0:
        config = ensure_subgraph_filtering_directory(config, "uni-umis", threshold)
        save_path = config.run_path
    else:
        return
    n_subgraph = 0
    for subgraph in subgraphs:
        n_subgraph +=1
        subgraph_edgelist = nx.to_pandas_edgelist(subgraph)
        subgraph_edgelist.to_csv(f"{save_path}/subgraph_{n_subgraph}_N={subgraph.number_of_nodes()}.csv", index = False)
        subgraph_edgelist.drop("weight", axis = 1).to_csv(f"{save_path}/subgraph_{n_subgraph}_N={subgraph.number_of_nodes()}_unw.csv", index = False)


def perform_subgraph_generation_by_filtering(config):
    config = generate_directory_names(config)
    convert_edgelist_format(config)
    # findSubgraphs(config)
    if config.subgraph_processing_args_filtering.bi_or_unipartite =="both":
        filter_bipartite(config)
        for threshold in config.subgraph_processing_args_filtering.nbead_filters_uni:
            filter_unipartite_beads(config, threshold)
        for threshold in config.subgraph_processing_args_filtering.numi_filters_uni:
            filter_unipartite_uMIs(config, threshold)
    elif config.subgraph_processing_args_filtering.bi_or_unipartite == "bi":
        filter_bipartite(config)
    elif config.subgraph_processing_args_filtering.bi_or_unipartite == "uni":
        if config.subgraph_processing_args_filtering.nbead_filters_uni != []:
            for threshold in config.subgraph_processing_args_filtering.nbead_filters_uni:
                filter_unipartite_beads(config, threshold)
        if config.subgraph_processing_args_filtering.numi_filters_uni != []:
            for threshold in config.subgraph_processing_args_filtering.numi_filters_uni:
                filter_unipartite_uMIs(config, threshold)

    else:
        print(f"Invalid input argument: {config.subgraph_processing_args_filtering.bi_or_unipartite} (should be 'uni', 'bi' , or 'both')")

if __name__== "__main__":
    from Utils import *

    config = ConfigLoader('config_standard_processes.py')

    perform_subgraph_generation_by_filtering(config)
