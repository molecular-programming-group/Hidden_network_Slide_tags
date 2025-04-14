import os
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import copy

class blank():
    def __init__(self):
        pass
class ConfigLoader:
    ''''
    This class takes a config file full of python dict structures and converts elements in dict and subdisct into attributes and subattributes instead for more readable config access
    '''
    def __init__(self, config_module, config_folder= "Configs"):
        # Import the config module dynamically
        if config_folder:
            # Add the folder to sys.path temporarily
            config_folder_path = Path(config_folder).resolve()
            if str(config_folder_path) not in sys.path:
                sys.path.insert(0, str(config_folder_path))

        try:
            # Import the config module dynamically
            config = importlib.import_module(config_module[:-3] if config_module.endswith('.py') else config_module)
        finally:
            if config_folder and str(config_folder_path) in sys.path:
                sys.path.remove(str(config_folder_path))  # Remove the folder from sys.path


        # Iterate over the attributes of the config module
        for attribute_name in dir(config):
            # Ignore built-in attributes
            if not attribute_name.startswith('__'):
                # Get the attribute value
                attribute_value = getattr(config, attribute_name)

                # If the attribute is a dictionary, convert it to an object recursively
                if isinstance(attribute_value, dict):
                    attribute_value = self._dict_to_object(attribute_value)

                # Set the attribute on the ConfigLoader instance
                setattr(self, attribute_name, attribute_value)
    def copy(self):
        """Returns a deep copy of the instance."""
        return copy.deepcopy(self)
    
    @staticmethod
    def _dict_to_object(data):
        """Convert a dictionary to an object where keys become attributes."""
        obj = type('ConfigObject', (object,), {})()
        for key, value in data.items():
            # Recursively convert dictionaries to objects
            if isinstance(value, dict):
                value = ConfigLoader._dict_to_object(value)
            setattr(obj, key, value)
        return obj
    
class Pointcloud(): #This class is mainly used to save and process the ground truth points generated from slidetag data in an easily accessible way
    def __init__(self, input_df, input_type="GT"):
        if input_type== "GT":
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

    def plot_points(self, ax=None, clr='blue', size = 10, alpha = 0.6, color_scheme=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # Default to square plot if standalone

        if color_scheme != None:
            clr = self.cell_types.map(color_scheme)
        # Scatter plot
        scatter = ax.scatter(self.x_coords, self.y_coords, s=size, c=clr, alpha=alpha)

        # Labels and title (adjust as needed)
        # ax.set_xlabel("X Coordinate")
        # ax.set_ylabel("Y Coordinate")
        # ax.set_title("Scatter Plot of X and Y Coordinates")
        
        # Set square aspect ratio for the plot
        ax.set_box_aspect(1)

        # Return the axes object for further use if needed
        
        return ax

    def printAttributes(self):
        # Print all attributes using vars()
        for attr, value in vars(self).items():
            print(f"{attr}: {value}")

# class Subgraph():
#     def __init__(self, filename = None, edgelist = None):
#         self.name = filename
#         self.original_edgelist = edgelist

def create_structure(): 

    ensure_directory("Input_files")
    ensure_directory("Intermediary_files")
    ensure_directory("Images")
    ensure_directory("Output_files")
    ensure_directory("STRND_structure")
    ensure_directory("Subgraph_edgelists")
    ensure_directory("Subgraph_reconstructions")
    ensure_directory("Configs")
    #region configs
    content_standard_processes = """# Its very important the sample name is exactly correct, as it is used to find filepaths in essentially every step
sample_name = "tonsil" #mouse_embryo, tonsil, mouse_hippocampus

preprocessing_args = {

    "filepaths" : {
                    "slidetags_output":             "df_whitelist_tonsil.txt", # df_whitelist_SRR11, df_whitelist_tonsil, df_whitelist_SRR07.txt
                    "barcodes_to_keep":             "barcodes_tonsil.csv",    # mouseembryo_barcodes, barcodes_tonsil, barcodes_hippocampus.csv
                    "barcodes_coordinates_file":    "HumanTonsil_spatial.csv", #mouseembryo_spatial, HumanTonsil_spatial, mousehippocampus_spatial.csv
                    "bc_exchange_path":             "3M-february-2018.txt",

                    "output_file":                  "all_cells.csv" # generally "only_spatial_cells.csv" or "all_cells.csv"
                    },
    
    "only_spatial_cells" : False
    }

filtering_args = {
    "use_custom_input"              :   False,               #if this is false, it will automatically use the preprocessing output as filtering input
                                                            # This will also be used in the subgraph_processing step, so it is required to got through here first
    "processed_edge_files"          :   "SRR07_edge_list_sequences.csv" #this hould be in the "Input_files" folder
                            , 
    "nUMI_sum_per_bead_thresholds"  :   [2, 1500], 
    "n_connected_cells_thresholds"  :   [2, 1500], 
    "per_edge_nUMI_thresholds"      :   1,    
    "include_N_beads"               :   False                    # Include or not include beads with an N in the sequence
    }

subgraph_processing_args_filtering = {
    "bi_or_unipartite"              :"bi",                         # "bi" or "uni" or "both"

    "minimum_subgraph_size"         :50               ,             #recommended to keep above 15 to ensure reconstruction quality 
    # !!!WARNING!!! If you change this, all reconstructions need be reconstructed or manually renamed again due to it also changing subgraph numbering
    
    "numi_filters_bi"               :[1],
    
    "nbead_filters_uni"             :[],
    "numi_filters_uni"              :[]

}

reconstruction_args = {
    "reconstruction_dimension"          : 2,
    "manual_reconstruction"             : False,                     #if True, Only the edgelist below will be reconstructed
    "manual_reconstruction_edgelist"    : "test.csv",            #full path of edgelist to reconstruct, it has to be in the right place in the STRND structure
    #-----------------------------
    "reconstructions_per_edgelist"      : 1,
    "reconstruct_all"                   : False,                      #if true will reconstruct all subgraphs of all filtering thresholds generated by this set of total configs
    "weighted_reconstruction"           : False,
    
    "bi_or_unipartite"                  : "bi",                         # "bi" or "uni" or "any", note bipartite only has the "umis" filter type
    "filter_type"                       : "umi",                    #"umis", "beads", or "any"
    
    "all_thresholds"                    : True ,                      
    "thresholds"                        : [],
    
    "which_subgraphs"                   : "all",    #"all", "biggest", or "specify". specify means the specific subgraph number, starts at 1
    "specific_subgraph_number"          : 3,
    "empty_strnd_folders"               : True           #This option if true deletes files in the edgelists and reconstructed positions in the STRND strucutre to prevent file confusions
}
    """
    content_subgraph_modification = '''sample_name = "tonsil" # mouse_embryo , tonsil

file_localization_args = {

    "ground_truth_file"   :"HumanTonsil_spatial.csv",  # HumanTonsil_spatial ,  mouseembryo_spatial
    "unfiltered_edge_file"     : "all_cells.csv", # generally "only_spatial_cells.csv" or "all_cells.csv"
    "run_parameters"           : {  
        "nUMI_sum_per_bead_thresholds"  :   [2, 1500], 
        "n_connected_cells_thresholds"  :   [2, 1500], 
        "per_edge_nUMI_thresholds"      :   1,    
                                    }

    }

# This set of configs controls the enriching where edges are taken from other edgelists of the same typ but at lower filtering values
simple_subgraph_enriching_args = {
    "subgraph_to_enrich" : {
    "bi_or_unipartite"                  : "bi",                        # "bi" or "uni" or "any", note bipartite only has the "umis" filter type
    "filter_type"                       : "umis",                    #"umis", "beads", or "any"
    "threshold"                        : [6, 7, 8, 9, 10],
    "enrich_all_subgraphs"               : True,             #True = enrich all subgraphs, False = choose specific below
    "subgraph_number"                   : 6
    
    },
    
    "enriching_sources" :{
        "use_same_type"                 : True,                
        "thresholds"                    : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],    
    },
    "reconstruct"                       : True,
    "reconstruction_dimension"          : 3,
    "reconstruct_n_times"               : 1,
    "empty_strnd_folders"               : True 
}

# This type of subgraph modification entails reconstructing the subgraph first, followed by removing edges above a certain length in the reconstruction
# and then re-reconstructing the subgraph
# NOTE: This requires having reconstructed the subgraph at least once and put it through cursory analysis which showed a population 
# of short, presumed spatially formed edges
perform_distance_gated_double_reconstruction ={
    "reconstruction_dimension"          : 2,
    "subgraph_to_enrich" : {
    "bi_or_unipartite"                  : "bi",                        # "bi" or "uni" or "any", note bipartite only has the "umis" filter type
    "filter_type"                       : "umis",                    #"umis", "beads", or "any"
    "threshold"                        : [1],
    "gate_percentiles"                   : [0.999],        #standard: 0.999, which percentile of the first gaussian should be used as the upper allowed limit of edge distances
    "recursive_gating"                  : False,        # if true perform the gating not only on base data but also edgeslists that have already been gated           
    "gate_all_subgraphs"               : True,             #True = gate all subgraphs, False = choose specific below
    "subgraph_number"                   : 6,
    },
    "re_reconstruction_args": {
            "reconstruction_dimension"          : 2,    
            "reconstruct_n_times"               : 1,   # Recommmended to be at least 5 for deeper analysis
            "empty_strnd_folders"               : True
    }
    
} 

perform_dbscan_edge_filtering = {
    "reconstruction_dimension"          : 2,
    "subgraph_to_enrich" : {
    "bi_or_unipartite"                  : "bi",                        # "bi" or "uni" or "any", note bipartite only has the "umis" filter type
    "filter_type"                       : "umis",
    "threshold"                        : [1],   
    "gate_all_subgraphs"               : True,             #True = gate all subgraphs, False = choose specific below
    "subgraph_number"                   : 6
    },   
    "dbscan_min_samples"                : [12],
    "dbscan_eps_percentages"             : [5],  
    "generate_pseudocells"              :True,
    "compensate_zero_clusters"          :False,    
    "re_reconstruction_args": {
            "reconstruction_dimension"          : 2,    
            "reconstruct_n_times"               : 1,   # Recommmended to be at least 5 for deeper analysis
            "empty_strnd_folders"               : True
    }
}'''

    content_subgraph_analysis = '''sample_name = "tonsil" # mouse_embryo , tonsil

base_network_args = {

    "ground_truth_file"                 : "HumanTonsil_spatial.csv", # HumanTonsil_spatial ,  mouseembryo_spatial
    "unfiltered_edge_file"              : "all_cells.csv", # only_spatial_cells, all_cells
    "run_parameters"           : {  
        "nUMI_sum_per_bead_thresholds"  :   [2, 1500], 
        "n_connected_cells_thresholds"  :   [2, 1500], 
        "per_edge_nUMI_thresholds"      :   1,    
                                    }

    }

predicted_cell_types_file = "tonsil_recon_gated_new_cell_predictions_metadata.csv"

filter_analysis_args = {
    "reconstruction_dimension"  : 2,
    "network_type"              : "bi", 
    "filter"                    : "umis",
    "analyse_all_thresholds"    : False,
    "thresholds_to_analyse"     : [1] # of not true, choose thresholds
}

subgraph_to_analyse = {
    "all_subgraphs"             : False       ,  
    "subgraph_number"           : 1       ,
    "knn_neighbours"            : 15         ,
    "minimum_subgraph_size"     : 10,
    "regenerate_detailed_edges" : False,
    "regenerate_summary_files"  : False,
    "gating_threshold"          : None, #"pseudo=False", #"pseudo=False", # None, all, dbscan, or a another indentifier as long as it is in the file name such as pseudo=False
    "include_recursively_gated" : False, 
    "include_ungated"           : True
    }

plot_modification = True
modification_type = "dbscan"  #gated, enriched, dbscan

vizualisation_args = {
    "how_many_reconstructions"  :1,                # number or "all" if there are multiple reconstructions, how many should be plotted
    "reconstruction_type"       : "recon", #recon, distortion, morphed_recon, or morphed_distortion
    "color_scheme"              : "cell_type",       # cell_type, vertical, horizontal, radius, knn, distortion, image
    "colormap"                  : "magma_r",     #Any matplotlib colormap, recommend viridis or tab10
    "colours"                   : "tonsil_2",

    "save_to_image_format"      : "png",
    "show_plots"                : True,               #having lots of plots can be memory limited, depending on their complexity
    "include_edges"             : False,
    "subsample_distortion"      : False
}
'''
    content_base_network_analysis = '''sample_name = "tonsil" # mouse_embryo , tonsil

base_network_args = {

    "ground_truth_file"        : "HumanTonsil_spatial.csv", # HumanTonsil_spatial ,  mouseembryo_spatial
    "unfiltered_edge_file"     : "all_cells.csv", # only_spatial_cells, all_cells
    "run_parameters"           : {  
        "nUMI_sum_per_bead_thresholds"  :   [2, 1500], 
        "n_connected_cells_thresholds"  :   [2, 1500], 
        "per_edge_nUMI_thresholds"      :   1,    
                                    }

    }

spatial_coherence_args = {
    "network_type"              :"bi",
    "filter_type"               :"umis",
    "min_subgraph_size_allowed" : 200
}
visualization_args = {
    "color_set"                     : "colors_mouse_embryo",
    "distance_analysis_thresholds"  : [1, 2, 3, 4, 5, 6, 7],
    "save_to_image_format"          : "png",
    "show_plots"                    :True
}'''
    #endregion
    for content, str in zip([content_standard_processes, content_subgraph_modification, content_subgraph_analysis, content_base_network_analysis], ["config_standard_processes", "config_subgraph_modification", "config_subgraph_analysis", "config_base_network_analysis"]):
        # Write content to the file
        if not os.path.isfile(f"Configs/{str}.py"):
            with open(f"Configs/{str}.py", 'w') as file:
                file.write(content)

def dictRandomcellColors(nucleus_coordinates):  # This functions generate a random color for each cell in the positional data
    import numpy as np
    cell_types = nucleus_coordinates["cell_type"][1:]
    type_to_colour_dict = {}
    for cell_type in cell_types.unique():
        color = list(np.random.choice(range(256), size=3))
        type_to_colour_dict[cell_type] = (color[0]/255,color[1]/255,color[2]/255)
    return type_to_colour_dict

def replace_first_folder(filepath, new_folder):
    from pathlib import Path
    # Convert the filepath to a Path object
    path = Path(filepath)
    
    # Reconstruct the path with the new folder as the first part
    new_path = Path(new_folder, *path.parts[1:])
    return str(new_path)  # Return as a string if needed

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_basic_barplot(values, identifiers, title="Basic Bar Plot", xlabel="Identifiers", ylabel="Values", color="skyblue", std_devs=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if std_devs:
        ax.bar(identifiers, values, yerr=std_devs, capsize=5, color=color, edgecolor='black')
    else:
        ax.bar(identifiers, values, capsize=5, color=color, edgecolor='black')

    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if title =="knn":
        ax.set_ylim([0,1.05])
    else:
        ax.set_ylim([0,1.05])
    if len(identifiers) ==3:
        ax.set_box_aspect(1)
    plt.tight_layout()
    

def generate_barcode_idx_mapping(config, edges_df):
    '''
    This function generates and mapping from barcode to index number for each unique bead and cell barcode in the largely unfiltered edgelist
    '''
    config = generate_directory_names(config)
    if os.path.isfile(f'{config.sample_path}/barcode_to_index_mapping_all.csv'):
        pass
    else:
        cell_bc_mapping = {bc: i for i, bc in enumerate(edges_df['cell_bc_10x'].unique(), start=1)} #start with assigning each cell a number ...

        start_index_bead = len(cell_bc_mapping) + 1
        bead_bc_mapping = {bc: i for i, bc in enumerate(edges_df['bead_bc'].unique(), start=start_index_bead)} #... then each bead as well

        barcode_to_index = {**cell_bc_mapping, **bead_bc_mapping} #Fuse the two dicts, cell and bead mapping to make one bif dict

        barcode_df = pd.DataFrame(list(barcode_to_index.items()), columns=['barcode', 'index']) # convert to df and the write to a csv for usage wherever relevant
        barcode_df.to_csv(f'{config.sample_path}/barcode_to_index_mapping_all.csv', index=False)


def generate_directory_names(config, usage_flag ="filtering"):
    '''
    As each set of filtering parameter has an indivudal folder, this functions simply acts as a simple way to get the correct names for findin the relevant files
    '''
    nUMI_thresholds = config.filtering_args.nUMI_sum_per_bead_thresholds
    n_connections_thresholds = config.filtering_args.n_connected_cells_thresholds
    per_edge_weight_threshold = config.filtering_args.per_edge_nUMI_thresholds

    
    if config.filtering_args.use_custom_input:
        config.filtering_directory = f"run={config.filtering_args.processed_edge_files[:-4]}_filters=numi{nUMI_thresholds[0]}-{nUMI_thresholds[1]}_nconn{n_connections_thresholds[0]}-{n_connections_thresholds[1]}_w{per_edge_weight_threshold}"
    else:
        config.filtering_directory = f"run={config.preprocessing_args.filepaths.output_file[:-4]}_filters=numi{nUMI_thresholds[0]}-{nUMI_thresholds[1]}_nconn{n_connections_thresholds[0]}-{n_connections_thresholds[1]}_w{per_edge_weight_threshold}"
    
    if usage_flag == "filtering":
        if config.filtering_args.use_custom_input ==False:
            config.sample_path = f"Intermediary_files/{config.sample_name}"
        else:
            config.sample_name = f"custom_edgelist/{config.filtering_args.processed_edge_files[:-4]}"
            config.sample_path = f"Intermediary_files/{config.sample_name}"
            

    return config

def ensure_subgraph_filtering_directory(config, filter_type="test", threshold=1, mode="save"):
    from Utils import ensure_directory, generate_directory_names
    config = generate_directory_names(config)
    #The "savepath" is one step deeper than the names generated by the "generateDirectoryNames" and is not used everywhere so is therefore generated by this function
    savepath = f"Subgraph_edgelists/{config.sample_name}/{config.filtering_directory}/{filter_type}_{threshold}"

    if mode =="search": #This is used to find all network filtering conditions for a specific uni or bipartite sample
        savepath = f"Subgraph_edgelists/{config.sample_name}/{config.filtering_directory}"
    ensure_directory(savepath)
    config.run_path = savepath
    return config
    
def runFullFilteringAndRawSubgraphPipeline(config):
    
    from initial_processing_functions import performPreprocessing
    performPreprocessing(config)
    from filtering_functions import performFiltering
    performFiltering(config)
    from subgraph_processing_functions import performSubgraphProcessing
    performSubgraphProcessing(config)
    from reconstruction_functions import interpretConfigAndReconstruct
    interpretConfigAndReconstruct(config)

if __name__ =="__main__":
    config = ConfigLoader('config_real.py')
    runFullFilteringAndRawSubgraphPipeline(config)