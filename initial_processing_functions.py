import pandas as pd

''' 
This code is for translating the raw output of the Russell et al code for sequencing data to compare it to the published cell positions, as the barcodes of the output does not match
as the barcodes are not directly comparable but need to be exchanged for its partner using the 10X provided list of approved barcode
which contains two barcodes per row, one of which matches the Russell et al determined positions and one which comes from the sequencing reads.

The input should be the output of Russel et al's 
'''

def load_data(filepaths_dict): # load the data types required

    #load the slide-tags output
    slidetags_output_edges = pd.read_table("Input_files/"+filepaths_dict.slidetags_output, sep = "\t")
    slidetags_output_edges.drop("CB_SB", axis = 1, inplace=True)                           
    slidetags_output_edges = slidetags_output_edges.reindex(columns=["cell_bc_10x", "bead_bc", "nUMI"])

    #Used if you want to specify which barcodes to keep, if using the slide-tags barcodes file nothing will change in the edge file since that is done with the slide-tags pipeline
    CR_barcodes = pd.read_csv("Input_files/"+filepaths_dict.barcodes_to_keep, header=None, names=["bc"])
    CR_barcodes = list(CR_barcodes["bc"])

    #For selecting only barcodes with a spatial position from slide-tags data
    barcodes_coordinates = pd.read_csv("Input_files/"+filepaths_dict.barcodes_coordinates_file)
    filtered_barcodes = list(barcodes_coordinates["NAME"][1:])

    #10X set of barcodes used to exchange its for its equivalent, since the output of the slide-tags sequencing processing is not the same as the one noted in the available spatial data
    exchange_df = pd.read_table("Input_files/"+filepaths_dict.bc_exchange_path,names=["bc_1", "bc_2"], sep="\t")
    return slidetags_output_edges, CR_barcodes, filtered_barcodes, exchange_df

def swap_paired_barcodes(raw_output_edgefile, exchange_df):
    # print(raw_output_edgefile)
    #Use a mapping dict for speed
    exchanged_edges = raw_output_edgefile.copy()
    bc_mapping = dict(zip(exchange_df["bc_2"], exchange_df["bc_1"]))

    # Create a new column with the updated cell barcodes using the mapping
    exchanged_edges['updated_cell_bc_10x'] = exchanged_edges['cell_bc_10x'].map(bc_mapping).fillna(raw_output_edgefile['cell_bc_10x'])

    exchanged_edges['cell_bc_10x'] = exchanged_edges['updated_cell_bc_10x']
    exchanged_edges.drop(columns=['updated_cell_bc_10x'], inplace=True)
    
    # print(raw_output_edgefile)
    # quit()
    return exchanged_edges

def filter_barcodes(All_edges_df, CR_barcodes, filtered_barcodes, only_spatial_cells=True):
    #For removing non-cellranger cells, as mentioned is already done but can be really be used to select of rany list of barcodes

    if only_spatial_cells:
        if filtered_barcodes[0][-2:] =="-1": #The Mouse Embryo has "-1" on the end of each barcode in the spatial file and this accounts for that 
            filtered_barcodes_set = set([barcode[:-2] for barcode in filtered_barcodes])
        else:
            filtered_barcodes_set = set([barcode for barcode in filtered_barcodes])
        Bad_cells_removed = All_edges_df[All_edges_df["cell_bc_10x"].isin(filtered_barcodes_set)]

    else:

        if CR_barcodes[0][-2:] =="-1": #The Mouse Embryo has "-1" on the end of each barcode in the spatial file and this accounts for that 
            CR_barcodes_set = set([barcode[:-2] for barcode in CR_barcodes])
        else:
            CR_barcodes_set = set([barcode for barcode in CR_barcodes])
        
        Bad_cells_removed = All_edges_df[All_edges_df["cell_bc_10x"].isin(CR_barcodes_set)]
    return Bad_cells_removed


def perform_preprocessing(config):

    #Master function that does each step  one after the other and saves the final output to a csv file
    from Utils import ensure_directory, generate_barcode_idx_mapping
    ensure_directory(f"Intermediary_files/{config.sample_name}")
    slidetags_output_edges, CR_barcodes, filtered_barcodes, exchange_df = load_data(config.preprocessing_args.filepaths)

    swapped_edgefile = swap_paired_barcodes(slidetags_output_edges, exchange_df)
    
    generate_barcode_idx_mapping(config, swapped_edgefile)
    Bad_cells_removed = filter_barcodes(swapped_edgefile, CR_barcodes, filtered_barcodes, config.preprocessing_args.only_spatial_cells)

    print("\nFiltered whitelist reindexed\n",Bad_cells_removed.reset_index(drop=True))
    Bad_cells_removed.to_csv(f"Intermediary_files/{config.sample_name}/{config.preprocessing_args.filepaths.output_file}", sep=",", index=False)

if __name__=="__main__":
    from initial_processing_functions import perform_preprocessing
    from Utils import *

    config = ConfigLoader('config_real.py')
    #print(config.preprocessing_args.filepaths.slidetags_output)  # Output: df_whitelist_SRR11.txt
    #print(config.preprocessing_args.use_only_filtered)           # Output: True

    #print(config.filtering_args.Sample)                          # Output: Mouse_Embryo
    #print(config.filtering_args.generate_plots)                  # Output: True
    #print(config.filtering_args.processed_edge_files[0])   
    perform_preprocessing(config)
