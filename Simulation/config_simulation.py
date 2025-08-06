
force_points_regeneration = False
reconstruct = "dev" #True, False, "dev" - if no reconstruction exists: reconstruction

base_on_sample = False
sample_file = "HumanTonsil_spatial.csv"  #mouseembryo_spatial.csv, mousehippocampus_spatial.csv, HumanTonsil_spatial.csv

diffusion = {
    "D"                 : [5, 10, 15, 20, 25],
    "times"              : [480],
    "noise_ratios"       : [0, 20, 30, 40, 45,47.5, 50, 52.5, 55, 60, 70, 75],                 # a percentage
    "edge_gen_mode"     :"pdf",          #pdf or powerlaw, both modes uses a powerlaw, but are slightly different in how it uses it to select edges
    "all_powerlaw_exp"      :[2]
}

# diffusion = {
#     "D"                 : [10],
#     "times"              : [480],
#     "noise_ratios"       : [40],                 # a percentage
#     "edge_gen_mode"     :"pdf",          #pdf or powerlaw, both modes uses a powerlaw, but are slightly different in how it uses it to select edges
#     "all_powerlaw_exp"      :[2]
# }

predefine_beads = False
bead_positions_file = ".csv"

predefine_cells = False
cell_positions_file = ".csv"

points = {
    "starting_cells":20000, 
    "generate_n_cells_by_cutting":False,
    "all_n_beads": [25000],
    "all_n_cells": [2500, 1250, 625, 250], 
    "bead_mode": "surface", # surface, cloud, surface2, lasagna
    "layers": 4                 #if mode == lasagna, 2 is the same as surface2
}

# points = {
#     "starting_cells":20000, 
#     "generate_n_cells_by_cutting":False,
#     "all_n_beads": [25000],
#     "all_n_cells": [625], 
#     "bead_mode": "surface", # surface, cloud, surface2, lasagna
#     "layers": 4                 #if mode == lasagna, 2 is the same as surface2
# }
space = {
    "shape"         : "cylinder",           # dome, rectangle, cylinder - round shapes use the x_span as diameter
    "dimension"     : 2 ,  
    "x_span"        : 1500, 
    "y_span"        : 20, 
    "z_span"        : 20, 

} 


reconstruction = {
    "recon_dimension"   :   2,
    "n_reconstructions" :   1

}