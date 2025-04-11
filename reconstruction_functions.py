
import os
import shutil
import pandas as pd
import re
from network_spatial_coherence.structure_and_args import GraphArgs
import network_spatial_coherence.nsc_pipeline as nsc
'''
These functions are for integrating into the pre-existing STRND structure as created by Fernandez Bonet et al. The actual reconstruction calculations are as origially published
'''
def initialize_paths(config):
    
    config.subgraph_path_split = config.edgelist_original_path.split("/")
    config.strnd_output_path = f"STRND_structure/{config.sample_name}/{config.filtering_directory}/{config.subgraph_path_split[-2]}"
    config.strnd_input_path = f"STRND_structure/data/edge_lists"
    config.strnd_reconstruction_path = f"STRND_structure/data/reconstructed_positions"
    config.reconstruction_path = f"Subgraph_reconstructions/{config.sample_name}/{config.filtering_directory}/{config.subgraph_path_split[-2]}_{config.reconstruction_args.reconstruction_dimension}D"
    print("input        ",config.edgelist_original_path)
    print("info output  ",config.strnd_output_path)
    print("strnd input  ",config.strnd_input_path)
    print("strnd recon  ",config.strnd_reconstruction_path)
    print("final output ",config.reconstruction_path)

    return config

def initalize_reconstruction(config, subgraph_name):

    args = GraphArgs(data_dir=f"{config.working_directory}/STRND_structure")
    # args = GraphArgs(data_dir=f"C:/Users/simon.kolmodin/Desktop/Slide_Tag/Publication_code/STRND_structure")

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
    args.edge_list_title = subgraph_name
    args.dim = config.reconstruction_args.reconstruction_dimension
    args.verbose = True

    return args

def perform_reconstruction(config, args):
    from Utils import ensure_directory
    graph, args = nsc.load_and_initialize_graph(args=args)
    single_graph_args, output_df = nsc.run_pipeline(graph, args)
    ensure_directory(config.strnd_output_path)
    output_df.to_csv(f"{config.strnd_output_path}/{config.subgraph_path_split[-1][:-4]}_recon_info.csv", index= False)
    output_df.set_index('Property', inplace=True)
    return output_df

def save_reconstruction(config, output_df):
    print(output_df)
    from Utils import ensure_directory
    reconstruction_file = [f for f in os.listdir(config.strnd_reconstruction_path) if f"N={output_df.at['num_points', 'Value']}" in f and "old_index" in f][0]
    ensure_directory(config.reconstruction_path)
    reconstruction_n = 1
    while True:
        if not os.path.isfile(f"{config.reconstruction_path}/{config.subgraph_path_split[-1][:-4]}_reconstruction_{reconstruction_n}.csv"):
            shutil.copy(f"{config.strnd_reconstruction_path}/{reconstruction_file}", f"{config.reconstruction_path}/{config.subgraph_path_split[-1][:-4]}_reconstruction_{reconstruction_n}.csv")
            break
        else:
            reconstruction_n += 1

def interpret_config_and_reconstruct(config, input_type="standard"):
    '''
    As the function has to find the files from previous step, reading and initialize the config is quite complex with many options, and therefore there are many conditions
    '''
    config.working_directory = os.getcwd() #This is for finding the correct output path for the STRND structure
    from Utils import ensure_subgraph_filtering_directory
    if config.reconstruction_args.manual_reconstruction:
        config = ensure_subgraph_filtering_directory(config)
    else:
        if input_type =="standard": #Files structure is slightly different if we want to reconstruct enriched subgraphs, as they are one folder "deeper"
            # config = initializePaths(config)
            config = ensure_subgraph_filtering_directory(config, mode="search")
            all_filters = [f for f in os.listdir(config.run_path)]
        elif input_type=="enriched" or input_type=="gated" or input_type=="dbscan":
            all_filters = [1]
            pass
        print(all_filters)
        if config.reconstruction_args.reconstruct_all:
            pass
        else:
            if config.reconstruction_args.bi_or_unipartite == "any":
                all_filters = [f for f in os.listdir(config.run_path)]
            else:
                all_filters = [f for f in os.listdir(config.run_path) if config.reconstruction_args.bi_or_unipartite in f]
            print(all_filters)
            if config.reconstruction_args.filter_type == "any":
                pass
            else:
                all_filters = [f for f in all_filters if config.reconstruction_args.filter_type in f]
            print(all_filters)
            if config.reconstruction_args.all_thresholds:
                pass
            else:
                all_filters = [f for f in all_filters if any(f"_{threshold}_" in f or f.endswith(f"_{threshold}") for threshold in config.reconstruction_args.thresholds)]        
        print(all_filters)
        for filter in all_filters:

            if input_type == "standard":
                subgraph_path = f"{config.run_path}/{filter}"
            elif input_type =="enriched" or input_type=="gated" or input_type=="dbscan":
                subgraph_path = f"{config.run_path}"

            if config.reconstruction_args.weighted_reconstruction:
                files = [f for f in os.listdir(subgraph_path) if os.path.isfile(os.path.join(subgraph_path, f)) and "unw" not in f]
            else:
                files = [f for f in os.listdir(subgraph_path) if os.path.isfile(os.path.join(subgraph_path, f)) and "unw" in f]

            if input_type =="gated" or input_type =="dbscan":
                files = [f for f in os.listdir(subgraph_path) if os.path.isfile(os.path.join(subgraph_path, f))]

            if config.reconstruction_args.which_subgraphs == "all":
                pass
            elif config.reconstruction_args.which_subgraphs == "biggest":
                # match = re.search(pattern, text)
                subgraph_sizes = [int(re.search(r"N=(\d+)", file).group(1)) for file in files]
                files = [file for file in files if int(re.search(r"N=(\d+)", file).group(1))==max(subgraph_sizes) or int(re.search(r"N=(\d+)", file).group(1)) == 55]
                if len(files)>1:
                    files = files[:1]
            elif config.reconstruction_args.which_subgraphs == "specify":
                subgraph_number = config.reconstruction_args.which_subgraphs
                files = [f for f in files if f"subgraph_{subgraph_number}_" in f]

            if input_type =="enriched":
                files = [f for f in files if any(f"w={threshold}." in f for threshold in config.simple_subgraph_enriching_args.enriching_sources.thresholds)] 
            
            for file in files:
                config.edgelist_original_path = f"{subgraph_path}/{file}"
                if input_type =="standard":
                    config = initialize_paths(config)
                elif input_type=="enriched" or input_type=="gated" or input_type=="dbscan":
                    config.subgraph_path_split = [file]
                    pass
                #initialize reconstruction arguments and copy edgelist from into the STRND structure for reconstruction
                for i in range(config.reconstruction_args.reconstructions_per_edgelist):

                    if config.reconstruction_args.manual_reconstruction ==True:
                        file = config.reconstruction_args.manual_reconstruction_edgelist
                        recon_args = initalize_reconstruction(file)
                    else:
                        recon_args = initalize_reconstruction(config, file)
                        shutil.copy(config.edgelist_original_path, config.strnd_input_path) # Has to be after reconstruction initialization to have the STRND structure present
                        # try:
                        output_df = perform_reconstruction(config, recon_args)
                        # except:
                        #     print("Reconstruction error, likely memory related")
                        save_reconstruction(config, output_df)

                        #This is to delete all the edgelists and reconstructions afterwards
                        #Everything is copied to other locations already and this prevents odd filename interactions
                        if config.reconstruction_args.empty_strnd_folders:
                            for filename in os.listdir(config.strnd_input_path):
                                file_path = os.path.join(config.strnd_input_path, filename)
                                if os.path.isfile(file_path):  # Check if it's a file
                                    os.remove(file_path)
                            for filename in os.listdir(config.strnd_reconstruction_path):
                                file_path = os.path.join(config.strnd_reconstruction_path, filename)
                                if os.path.isfile(file_path):  # Check if it's a file
                                    os.remove(file_path)

if __name__== "__main__":

    from Utils import *

    config = ConfigLoader('config_standard_processes.py')
    interpret_config_and_reconstruct(config)


