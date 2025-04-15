import pandas as pd
import os
import numpy as np


def umi_sum_filtering(config):
    '''
    This first functions removes all beads with a total number of UMIs outiside of the set lower and upper thresholds.
    It is very fast
    '''
    if config.filtering_args.use_custom_input:
        edge_list = pd.read_csv(f"Input_files/{config.filtering_args.processed_edge_files}", index_col = "Unnamed: 0")
    else:
        edge_list = pd.read_csv(f"{config.sample_path}/{config.preprocessing_args.filepaths.output_file}")

    #This gives each cell and bead a number, this is used for later edgelists instead of whole barcodes
    from Utils import generate_barcode_idx_mapping
    generate_barcode_idx_mapping(config, edge_list)

    edge_list_abundant_beads_cut = edge_list.set_index("bead_bc", drop = True)

    #extract paths and threshold from config
    nUMI_thresholds = config.filtering_args.nUMI_sum_per_bead_thresholds
    filename_numi_sum = f"{config.sample_path}/{config.filtering_directory}/{config.sample_name}_nUMI_sum_filter_{nUMI_thresholds[0]}_{nUMI_thresholds[1]}.csv"
    config.filename_numi_sum = filename_numi_sum

    if os.path.isfile(filename_numi_sum): # if this exact file is already made, load it instead of remaking it
        print(f"Found file:\n   {filename_numi_sum}, \nloading in place of re-filtering")
        return pd.read_csv(filename_numi_sum)
    
    #Make a new edgelist grouped by the bead BC, summing the UMIs
    filtering_df = edge_list_abundant_beads_cut.groupby("bead_bc").sum()
    beads_to_cut = [        #Go thorugh the summed df, marking beads which does not fall within the thresholds
        bead for bead, nUMI_sum_collapsed in filtering_df.iterrows()
        if "N" in bead and not config.filtering_args.include_N_beads or nUMI_sum_collapsed['nUMI'] > nUMI_thresholds[1] or nUMI_sum_collapsed['nUMI'] < nUMI_thresholds[0]
    ]   
    #In the un-grouped edgelist, find all beads marked previously and only keep edges that does not have any of those beads
    edge_list_abundant_beads_cut = edge_list_abundant_beads_cut[~edge_list_abundant_beads_cut.index.isin(beads_to_cut)]
    edge_list_abundant_beads_cut.to_csv(f"{config.sample_path}/{config.filtering_directory}/{config.sample_name}_nUMI_sum_filter_{nUMI_thresholds[0]}_{nUMI_thresholds[1]}.csv")
    
    return edge_list_abundant_beads_cut

def n_connections_filtering(config, edge_list = None):
    '''
    This first functions removes all beads connencting to a number of different cells outiside of the set lower and upper thresholds.
    It works much the same way, except it counts how many edges each bead has using np.unique() and uses that too cut 
    It is quite fast
    '''
    if edge_list is None:
        edge_list = pd.read_csv(config.filename_numi_sum)
    n_connections_thresholds = config.filtering_args.n_connected_cells_thresholds
    filename_n_connections = f"{config.sample_path}/{config.filtering_directory}/{config.sample_name}_n_connections_filter_{n_connections_thresholds[0]}-{n_connections_thresholds[1]}.csv"
    config.filename_n_connections = filename_n_connections
    if os.path.isfile(filename_n_connections):
        print(f"Found file:\n   {filename_n_connections},\nloading in place of re-filtering")
        return pd.read_csv(filename_n_connections)
    all_beads_post_sum_filter = np.unique(edge_list["bead_bc"], return_counts=True)
    beads_to_cut = [
        bead for bead, count in zip(all_beads_post_sum_filter[0], all_beads_post_sum_filter[1])
        if count < n_connections_thresholds[0] #check if its lower than the lower threshold 
        or count > n_connections_thresholds[1] #or higher than the upper threshold
    ]

    edge_list_filtered = edge_list[~edge_list["bead_bc"].isin(beads_to_cut)]
    edge_list_filtered.to_csv(filename_n_connections, index=False)
    
    return edge_list_filtered

def per_edge_umi_filtering(config, edge_list = None):
    '''
    This final filtering simply cuts each cell-bead edge and cuts it if it is made up of less than the chosen number of UMIs, so if this threhold is 1 it does nothing
    It is very fast
    '''
    per_edge_weight_threshold = config.filtering_args.per_edge_nUMI_thresholds
    filename_per_edge_weight = f"{config.sample_path}/{config.filtering_directory}/{config.sample_name}_edge_list_filtered_by_per_edge_weight_{per_edge_weight_threshold}.csv"
    if edge_list is None:
        edge_list = pd.read_csv(config.filename_n_connections)
    config.filename_per_edge_weight = filename_per_edge_weight
    if os.path.isfile(filename_per_edge_weight):
        print(f"Found file:\n   {filename_per_edge_weight}, \nloading in place of re-filtering")
        return pd.read_csv(filename_per_edge_weight)

    edge_list_filtered = edge_list[edge_list["nUMI"] >= per_edge_weight_threshold].reset_index(drop=True)
    edge_list_filtered.to_csv(filename_per_edge_weight, index=False)
    
    return edge_list_filtered

def generate_bead_cell_weight_matrix(config, edge_list=None):
    '''
    This part generates a bead-cell adjacency matrix from the edgelist generated by the last step in the filtering. It is quite slow, due to the many elements required for a full adjacency matrix and can be susceptible to RAM limitations
    '''
    print("Generating bead-cell adjacency matrix")

    config.filename_cell_bead_mtx = f"{config.sample_path}/{config.filtering_directory}/bead-cell_weight_matrix.csv"
    if os.path.isfile(config.filename_cell_bead_mtx):
        print(f"Found file:\n   {config.filename_cell_bead_mtx}, \nloading in place of re-generating matrix")
        return pd.read_csv(config.filename_cell_bead_mtx, index_col=0)
    if edge_list is None:
        edge_list = pd.read_csv(config.filename_per_edge_weight)

    #First, generate the empty dataframe with beads as rows and cells as columns
    unique_bead_bcs = edge_list["bead_bc"].unique()
    unique_cell_bcs = edge_list["cell_bc_10x"].unique()
    weights_bead_to_cells = pd.DataFrame(0, index=unique_bead_bcs, columns=unique_cell_bcs)

    #Then, for each row bead-cell edge, set the corresponding element in the adjacency matrix to the UMIs (weight) of the edge in the edgelist
    for _, row in edge_list.iterrows():
        weights_bead_to_cells.at[row['bead_bc'], row['cell_bc_10x']] = row['nUMI']
    print("Saving to CSV")
    weights_bead_to_cells.to_csv(config.filename_cell_bead_mtx)

    return weights_bead_to_cells

def convert_bead_cell_to_cell_cell(config, weights_bead_to_cells=None):
    '''
    This step takes the bead-cell adjacency matrix and converts it to a unipartite cell-cell adjacency matrix. It has two sets of weight metrics, the number of unique beads connect each cell
    and the total sum of UMIs connecting the cells. It is quite slow, and also susceptible to RAM limitations to the bead-cell matrix' size
    Both of these functions does however only need to be done once for each set of filtering parameters and input edgelist
    '''
    
    config.filename_cell_cell_beads_mtx = f"{config.sample_path}/{config.filtering_directory}/cell-cell_beadweight_matrix.csv"
    config.filename_cell_cell_umis_mtx = f"{config.sample_path}/{config.filtering_directory}/cell-cell_umiweight_matrix.csv"
    
    if os.path.isfile(config.filename_cell_bead_mtx) and os.path.isfile(config.filename_cell_cell_umis_mtx):
        print(f"Found files \n  {config.filename_cell_bead_mtx}\nand\n  {config.filename_cell_cell_beads_mtx}, \nloading in place of re-generating matrix")
        weights_cells_to_cells = pd.read_csv(config.filename_cell_cell_umis_mtx, index_col=0)
        weights_n_beads_cells_to_cells = pd.read_csv(config.filename_cell_cell_beads_mtx, index_col=0)
        return weights_cells_to_cells, weights_n_beads_cells_to_cells
    if weights_bead_to_cells is None:
        weights_bead_to_cells = pd.read_csv(config.filename_cell_bead_mtx, index_col=0)
    
    
    print("Generating cell-cell adjacency matrices with UMI and Bead weight metrics")
    weights = weights_bead_to_cells.to_numpy()
    cells = weights_bead_to_cells.columns

    # Precompute pairwise contributions and bead counts
    num_cells = len(cells)
    #Initiate two empty dataframes for each weight metric
    umi_contributions = np.zeros((num_cells, num_cells), dtype=np.float32)
    bead_counts = np.zeros((num_cells, num_cells), dtype=np.int32)

    # Iterate over rows (beads)
    for bead_weights in weights:
        # Find connected cells for the current bead
        connected_indices = np.nonzero(bead_weights)[0]
        if len(connected_indices) > 1: # These will generally also be removed by the previous filtering steps
            # Create all pairwise combinations of connected cells
            idx_pairs = np.array(np.meshgrid(connected_indices, connected_indices)).T.reshape(-1, 2)
            
            # Avoid self-pairs
            idx_pairs = idx_pairs[idx_pairs[:, 0] != idx_pairs[:, 1]]

            # Update UMI contributions and bead counts
            for cell1_idx, cell2_idx in idx_pairs:
                umi_contributions[cell1_idx, cell2_idx] += bead_weights[cell1_idx] + bead_weights[cell2_idx]
                bead_counts[cell1_idx, cell2_idx] += 1
    weights_cells_to_cells = pd.DataFrame(umi_contributions, index=cells, columns=cells)
    weights_n_beads_cells_to_cells = pd.DataFrame(bead_counts, index=cells, columns=cells)

    weights_cells_to_cells.to_csv(config.filename_cell_cell_umis_mtx)
    weights_n_beads_cells_to_cells.to_csv(config.filename_cell_cell_beads_mtx)
    print("Saving to CSV")
    return weights_cells_to_cells, weights_n_beads_cells_to_cells

def perform_filtering(config):
    #master functions that does everything in order
    from Utils import ensure_directory, generate_directory_names

    config = generate_directory_names(config)

    ensure_directory(f"{config.sample_path}/{config.filtering_directory}")

    edgelist = umi_sum_filtering(config)

    print(edgelist)
    edgelist = n_connections_filtering(config)
    print(edgelist)
    edgelist = per_edge_umi_filtering(config)
    print(edgelist)
    # mtx = generate_bead_cell_weight_matrix(config, edgelist)
    # print(mtx)
    # mtx1, mtx2 = convert_bead_cell_to_cell_cell(config, mtx)
    # print(mtx1)
    # print(mtx2)

if __name__ =="__main__":
    from filtering_functions import perform_filtering
    from Utils import *

    config = ConfigLoader('config_standard_processes.py')
    perform_filtering(config)