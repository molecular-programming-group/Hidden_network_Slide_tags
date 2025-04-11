import pandas as pd

''' 
This code is for translating the raw output of the Russell et al code for sequencing data to compare it to the published cell positions, as the barcodes of the output does not match
as the barcodes are not directly comparable but need to be exchanged for its partner using the 10X provided list of approved barcode
which contains two barcodes per row, one of which matches the Russell et al determined positions and one which comes from the sequencing reads.

it alsow removed edges connecting having at least one edge not 
'''

def load_data(filepaths_dict): # load the data types required

    #load the slide-tags output
    slidetags_output_edges = pd.read_table(filepaths_dict["slidetags_output"], sep = "\t")
    slidetags_output_edges.drop("CB_SB", axis = 1, inplace=True)                           
    slidetags_output_edges = slidetags_output_edges.reindex(columns=["cell_bc_10x", "bead_bc", "nUMI"])

    #Used if you want to specify which barcodes to keep, if using the slide-tags barcodes file nothing will change in the edge file since that is done with the slide-tags pipeline
    CR_barcodes = pd.read_csv(filepaths_dict["cr_barcodes_file"], header=None, names=["barcode"])

    #For selecting only barcodes with a spatial position from slide-tags data
    barcodes_coordinates = pd.read_csv(filepaths_dict["barcodes_coordinates_file"])
    filtered_barcodes = list(barcodes_coordinates["NAME"][1:])
    
    #10X set of barcodes used to exchange its for its equivalent, since the output of the slide-tags sequencing processing is not the same as the one noted in the available spatial data
    exchange_df = pd.read_table(filepaths_dict["bc_exchange_path"],names=["bc_1", "bc_2"], sep="\t")
    return slidetags_output_edges, CR_barcodes, filtered_barcodes, exchange_df

def swapPairedBarcodes(raw_output_edgefile, exchange_df):
    # print(raw_output_edgefile)
    #Use a mapping dict for speed
    exchanged_edges = raw_output_edgefile.copy()
    bc_mapping = dict(zip(exchange_df["bc_2"], exchange_df["bc_1"]))

    # Create a new column with the updated cell barcodes using the mapping
    exchanged_edges['updated_cell_bc_10x'] = exchanged_edges['cell_bc_10x'].map(bc_mapping).fillna(raw_output_edgefile['cell_bc_10x'])

    # If inplace replacement of the original column is needed
    exchanged_edges['cell_bc_10x'] = exchanged_edges['updated_cell_bc_10x']
    exchanged_edges.drop(columns=['updated_cell_bc_10x'], inplace=True)
    
    # print(raw_output_edgefile)
    # quit()
    return exchanged_edges

def filter_barcodes(All_edges_df, CR_barcodes, filtered_barcodes, use_only_filtered=True):
    #For removing non-cellranger cells, as mentioned is already done but can be really any list of barcodes

    if use_only_filtered:
        if filtered_barcodes[0][-2:] =="-1": #The Mouse Embryo has "-1" on the end of each barcode in the spatial file and this accounts for that 
            filtered_barcodes_set = set([barcode[:-2] for barcode in filtered_barcodes])
        else:
            filtered_barcodes_set = set([barcode for barcode in filtered_barcodes])
        Bad_cells_removed = All_edges_df[All_edges_df["cell_bc_10x"].isin(filtered_barcodes_set)]

    else:
        CR_barcodes_set = set(CR_barcodes["barcode"].values)
        Bad_cells_removed = All_edges_df[All_edges_df["cell_bc_10x"].isin(CR_barcodes_set)]

    return Bad_cells_removed

def main():

    filepaths_dict = {
        "slidetags_output": "SRR11/df_whitelist_SRR11.txt", 
        "cr_barcodes_file": "Mouse_Embryo/barcodes_ncbi.csv", 
        "barcodes_coordinates_file": "Mouse_Embryo/mouseembryo_spatial.csv",
        "bc_exchange_path": "3M-february-2018.txt",

        "output_file": "Mouse_Embryo_edge_list_sequences_only_spatial_cells.csv"
    }
    slidetags_output_edges, CR_barcodes, filtered_barcodes, exchange_df = load_data(filepaths_dict)

    swapped_edgefile = swapPairedBarcodes(slidetags_output_edges, exchange_df)
    use_only_filtered = True

    Bad_cells_removed = filter_barcodes(swapped_edgefile, CR_barcodes, filtered_barcodes, use_only_filtered)

    print("filtered whitelist\n", Bad_cells_removed)
    print(Bad_cells_removed.reset_index(drop=True))
    Bad_cells_removed.reset_index(drop=True).to_csv(filepaths_dict["output_file"], sep = ",")

if __name__ == "__main__":
    main()

# filtered_df.to_csv("10X_bc_to_cell_bc_SRR11.csv", index = False)