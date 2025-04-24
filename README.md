__More updates coming soon, including biological analysis, input files, bugfixing, figure generation, ease of use, and annotation__

# How to setup the environment and run a first-pass reconstruction

As the first step after aquiring the scripts in this repository is to set upthe environment
While most of the code operates on relatively standard python packages (numpy, pandas, matplotlib etc.) packages related to the STRND reconstruction (available [HERE](https://github.com/DavidFernandezBonet/Network_Spatial_Coherence)) and the reconstruction "morphing" (available [HERE](https://github.com/DavidFernandezBonet/alphamorph)) has to be installed. 
To run the STRND package 3.11 or 3.12 is recommended, python 3.11 was used for this manuscript.

The .yml file can be used to create the required environment using Conda

```bash 
conda env create -f slide_tags_network.yml
```

Alternatively, the packages required will also be installed as a part of these three packages: 

```bash
pip install network_spatial_coherence
pip install alphamorph
pip install notebook
```

After setting up the environment, the inital processing and a first-pass reconstruction can be performed using the "Preprocessing.ipynb" notebook.
Alternatively, the individual scripts can be run individually and the the order to run is:
1. Utils.py
2. initital_processing_functions.py
3. filtering_functions.py
4. reconstruction_functions.py
And for reconstruction analysis then run "sugbraph_analysis_functions.py"

This there are multiple options for further analysis:
1. Perform more in-depth analysis on the reconstruction with many options in the "additional_subgraph_analysis" notebook
2. Perform an iterative reconstruction in the "subgraph_modification" notebook
3. Perform the biological analysis using the R-based "slidetags-network" R project

To asess non-reconstruction related network properties, the "Base_network_analysis" notebook allows analysis of degree distributions etc. for multiple sample
