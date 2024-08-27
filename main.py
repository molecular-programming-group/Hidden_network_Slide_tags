from Publication_functions import *

import os
import pandas as pd
from scipy.stats import expon, lognorm, weibull_min

def generateResampledEdgeFiles(args):
    Sample                      = args["Sample"]
    Sample_run                  = args["Sample_run"]
    edgelist_to_refilter        = args["edgelist_to_refilter"]
    edgelist_to_refilter_list   = args["edgelist_to_refilter_list"]
    extra_edges_filename        = args["extra_edges_filename"]

    if args["one_or_all"]==1:
        edgelist_to_refilter_list = [edgelist_to_refilter]
    
    for edgelist_to_refilter in edgelist_to_refilter_list:

        original_edgelist_path = f"{Sample}/{Sample_run}/data/edge_lists/{edgelist_to_refilter}"
        if args["refilter"] ==1:
            if type(extra_edges_filename) ==list:
                for extra_edge_file in extra_edges_filename:
                    extra_edges_path = f"{Sample}/{Sample_run}/data/edge_lists/{extra_edge_file}"
                    refilteredEdgelistGeneration(original_edgelist_path, extra_edges_path)
                    print(extra_edges_path)
            else:
                extra_edges_path = f"{Sample}/{Sample_run}/data/edge_lists/{extra_edges_filename}"
                refilteredEdgelistGeneration(original_edgelist_path, extra_edges_path)

def analyzeSubgraphImprovementDavidOutput(args):
    Sample                      = args["Sample"]
    Sample_run                  = args["Sample_run"]
    color_type                  = args["color_type"]
    cell_to_color_dict          = args["cell_to_color_dict"]
    dilation_scale              = args["dilation_scale"]
    edgelist_to_refilter        =  args["edgelist_to_refilter"]
    edgelist_to_refilter_list   = args["edgelist_to_refilter_list"]
    nucleus_coordinates = args["all_true_coordinates"]
    seq_to_node = pd.read_csv(f"{Sample}/{Sample}_cell_and_bead-idx_mapping.csv", names=["cell", "node_ID"]).set_index("node_ID")
    original_edgelist_path = f"{Sample}/{Sample_run}/data/edge_lists/{edgelist_to_refilter}"

    if args["one_or_all"]==1:
        edgelist_to_refilter_list = [edgelist_to_refilter]

    seq_to_node = pd.read_csv(f"{Sample}/{Sample}_cell_and_bead-idx_mapping.csv", names=["cell", "node_ID"]).set_index("node_ID")


    print(nucleus_coordinates)
    print(seq_to_node)


    all_reconstruction_files = os.listdir(f"{Sample}/{Sample_run}/data/reconstructed_positions/")

    if args["plot_improved_subgraphs"] == True:
        all_subgraphs = multi_subgraph_class()
        for edgelist_to_refilter in edgelist_to_refilter_list:
            original_edgelist_path = f"{Sample}/{Sample_run}/data/edge_lists/{edgelist_to_refilter}"
            all_subgraph_reconstruction_paths_and_refiltered = [f"{Sample}/{Sample_run}/data/reconstructed_positions/{reconstruction}" for reconstruction in all_reconstruction_files if edgelist_to_refilter.split("N=")[-1][:-4] in reconstruction and "positions_old_index" in reconstruction]
            
            for str in all_subgraph_reconstruction_paths_and_refiltered:
                print(str)
                count = str.count("nbead")
                print(count)
            
            if args["which_reconstructions"] == "refilter":
                relevant_reconstructions = [path for path in all_subgraph_reconstruction_paths_and_refiltered if "extension" not in path]
                for file in relevant_reconstructions:
                    print(file)
                    print(original_edgelist_path)

                if len(relevant_reconstructions)==0:
                    print("No relevant reconstructions found, continuing")
                    continue
                concatenated_subgraph = plotAllRefilteredSubgraphs(nucleus_coordinates, seq_to_node,original_edgelist_path, relevant_reconstructions, args)            
            all_subgraphs.add_subgraph(concatenated_subgraph)
    return all_subgraphs 
    
def distortionHistograms(subgraphs, args):
    Sample                      = args["Sample"]
    Sample_run                  = args["Sample_run"]
    for subgraph in subgraphs.all_subgraphs:
        distortions = subgraph.get_second_to_last_subgraph().distortion_all_points
        bin_width = 5

        # Calculate KDE
        kde = gaussian_kde(distortions, bw_method='scott')
        kde_x = np.linspace(0, max(distortions) + bin_width, 1000)
        kde_y = kde(kde_x)

        # Plot the one-sided violin plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Fit log-normal distribution and plot
        params_lognorm = lognorm.fit(distortions)
        pdf_lognorm = lognorm.pdf(kde_x, *params_lognorm)
        mean_lognorm = lognorm.median(*params_lognorm)
        median = np.median(distortions)
        mean = np.mean(distortions)
        peak_lognorm = kde_x[np.argmax(pdf_lognorm)]
        for point in distortions:
            ax.axvline(x=point, ymin=0, ymax=0.02, color='grey', alpha=0.5, linewidth=0.5)

        plt.fill_betweenx(kde_y,kde_x, alpha=0.5)
        plt.plot(kde_x, pdf_lognorm, label='Fitted Log-Normal', color='orange')
        plt.axvline(x=peak_lognorm, color='red', linestyle='--', label=f'Peak (Log-Normal) = {peak_lognorm:.2f} µm')
        plt.axvline(x=mean_lognorm, color='green', linestyle='--', label=f'Mean (Log-Normal) = {mean_lognorm:.2f} µm')
        plt.axvline(x=median, color='orange', linestyle='--', label=f'Median = {median:.2f} µm')
        plt.axvline(x=mean, color='yellow', linestyle='--', label=f'mean = {mean:.2f} µm')

        plt.xlabel('Distortion (µm)')
        ax.set_box_aspect(1)
        ax.set_ylim([0, 0.02])
        plt.ylabel('Density')
        plt.title('One-Sided Violin Plot with Custom Bin Width')
        plt.legend()

        plt.savefig(f"{Sample}/{Sample_run}/distortion_N={len(subgraph.get_second_to_last_subgraph().colors_points)}.pdf", format="pdf")
        plt.show()
    pass

#region setting parameters

Sample = "Mouse_Embryo"
Sample_run = "20240516_spatial_bead_sum1-256_edge_1_n_connections_1-256"

entries = os.listdir(Sample+"/") 
str_id = "spatial"
spatial_file = [entries for entries in entries if str_id in entries][-1]
nucleus_coordinates_df = pd.read_csv(Sample+"/"+spatial_file).set_index("NAME")

color_dict = dictRandomcellColors(nucleus_coordinates_df)
color_dict = {
    'Neuron_1': (0.6992848904267589, 0.7927873894655901, 0.8838754325259516, 1.0),
    'Neuron_2': (0.650565167243368, 0.7588312187620146, 0.866159169550173, 1.0),
    'Neuron_3': (0.609919261822376, 0.71680123029604, 0.8441368704344483, 1.0),
    'Neuron_4': (0.5833448673587083, 0.6606997308727413, 0.8146097654748173, 1.0),
    'Neuron_5': (0.5567704728950404, 0.6045982314494426, 0.7850826605151865, 1.0),
    'Neuron_6': (0.5490196078431373, 0.5432679738562092, 0.7545098039215686, 1.0),
    'Neuron_7': (0.5490196078431373, 0.47978469819300285, 0.7235063437139563, 1.0),
    'Neuron_8': (0.5487120338331412, 0.4163783160322953, 0.6925797770088428, 1.0),
    'Neuron_9': (0.5428066128412149, 0.3543713956170704, 0.6630526720492118, 1.0),
    'Neuron_10': (0.5369011918492888, 0.2923644752018454, 0.633525567089581, 1.0),
    'Neuron_11': (0.5292425990003845, 0.22568242983467912, 0.5964013840830451, 1.0),
    'Neuron_12': (0.5189081122645136, 0.15186466743560167, 0.547681660899654, 1.0),
    'Neuron_13': (0.5085736255286428, 0.07804690503652442, 0.49896193771626296, 1.0),
    'Neuron_14': (0.4491041906958862, 0.04244521337946944, 0.43277201076509036, 1.0),
    'Neuron_15': (0.372333717800846, 0.020299884659746303, 0.3604306036139948, 1.0),
    'Neuron_16': (0.30196078431372547, 0.0, 0.29411764705882354, 1.0),
    'Choroid Plexus': (0.9969242599000384, 0.9224913494809689, 0.8147635524798155, 1.0),
    'Endothelial': (0.9921568627450981, 0.772164552095348, 0.5580315263360246, 1.0),
    'Fibroblast': (0.968242983467897, 0.4914263744713572, 0.322875816993464, 1.0),
    'White Blood Cells': (0.8132410611303345, 0.14837370242214531, 0.09582468281430219, 1.0),
    'Radial glia': (0.4980392156862745, 0.0, 0.0, 1.0)}

color_dict_collapsed = {
    'Choroid Plexus': (0.31045520848595526, 0.07845687167618218, 0.2660200572779356, 1.0),
    'Endothelial': (0.8857501584075443, 0.8500092494306783, 0.8879736506427196, 1.0),
    'Fibroblast': (0.3478165323877778, 0.1573386351115298, 0.5511085345270326, 1.0),
    'Neuron': (0.4449324685421538, 0.5885055998308617, 0.7551311041857901, 1.0),
    'Radial glia': (0.7105456546582587, 0.36065500290840113, 0.3275357416278876, 1.0),
    'White Blood Cells': (0.8594346370300241, 0.8014392309950078, 0.7843978875138392, 1.0)}

resample_args = {
"Sample"                    : Sample,
"Sample_run"              : Sample_run,
"refilter"                : 1,
"add_points_subgraphs"    : 0,
"one_or_all"              : 1,
"edgelist_to_refilter"      :  "old_index_edge_list_N=3701_dim=2_experimental_edge_list_nbead_4_filtering.csv",
"edgelist_to_refilter_list"     : ["old_index_edge_list_N=286_dim=2_experimental_edge_list_nbead_7_filtering_component_11.csv",
                            "old_index_edge_list_N=204_dim=2_experimental_edge_list_nbead_7_filtering_component_9.csv", 
                            "old_index_edge_list_N=149_dim=2_experimental_edge_list_nbead_7_filtering_component_4.csv",
                            "old_index_edge_list_N=142_dim=2_experimental_edge_list_nbead_7_filtering_component_34.csv",
                            "old_index_edge_list_N=48_dim=2_experimental_edge_list_nbead_7_filtering_component_13.csv"],
"extra_edges_filename"        : [
    
                            "edge_list_nbead_4_filtering.csv",
                            "edge_list_nbead_3_filtering.csv", 
                            "edge_list_nbead_2_filtering.csv",
                            "edge_list_nbead_1_filtering.csv"]
}



args = {
    "plot_KNN"              :False,
    "plot"                  :True,
    "save_plots"            :False,
    "save_vector_file"      :False  ,
"reconstruction"            : "node2vec",
"Sample"                    : Sample,
"Sample_run"              : Sample_run,
"plot_improved_subgraphs" : True,
"one_or_all"              : 1,
"which_reconstructions"   : "refilter",             #"refilter", "extension" # Types of subgraph enrichment, extensions remains unused
"color_type"              : "cell" ,                #"cell" for per cell type, "gray" for grayscale based on x coordinate
"all_true_coordinates"    : nucleus_coordinates_df,
"cell_to_color_dict"      : color_dict,
"collapse_neurons"        : False,
"dilation_scale"          : 1.2,
"edgelist_to_refilter"      :  "old_index_edge_list_N=286_dim=2_experimental_edge_list_nbead_7_filtering_component_11.csv",
#cell_network_edges_lambda400_converted.csv
"edgelist_to_refilter_list"     : ["old_index_edge_list_N=286_dim=2_experimental_edge_list_nbead_7_filtering_component_11.csv",
                            "old_index_edge_list_N=204_dim=2_experimental_edge_list_nbead_7_filtering_component_9.csv", 
                            "old_index_edge_list_N=149_dim=2_experimental_edge_list_nbead_7_filtering_component_4.csv",
                            "old_index_edge_list_N=142_dim=2_experimental_edge_list_nbead_7_filtering_component_34.csv",
                            "old_index_edge_list_N=48_dim=2_experimental_edge_list_nbead_7_filtering_component_13.csv"],
"extra_edges_filename"        : ["edge_list_nbead_9_filtering.csv",
                            "edge_list_nbead_8_filtering.csv",
                            "edge_list_nbead_7_filtering.csv",
                            "edge_list_nbead_6_filtering.csv", 
                            "edge_list_nbead_5_filtering.csv", 
                            "edge_list_nbead_4_filtering.csv",
                            "edge_list_nbead_3_filtering.csv", 
                            "edge_list_nbead_2_filtering.csv",
                            "edge_list_nbead_1_filtering.csv"]
}
if args["collapse_neurons"] == True:
    args["cell_to_color_dict"] = color_dict_collapsed
all_subgraphs = analyzeSubgraphImprovementDavidOutput(args)
# generateResampledEdgeFiles(resample_args)
# DrawKNNandCPD(all_subgraphs, args)
distortionHistograms(all_subgraphs, args)
# violinKNN(all_subgraphs, args)
# barDensity(all_subgraphs, args)
for subgraph in all_subgraphs.all_subgraphs:
    subgraph.make_position_file()
# plotDensity(all_subgraphs)
plt.show()