__More updates coming soon, including input files, bugfixing, figure generation, ease of use, and annotation__

In this repository there are notebooks that give more detailed instructions on how to run the specific workflow and functions. Most scripts can also be ran by themselves, although its recommended to instead import the functions for running the scripts since that allows better handling of the configurations files.
there are several config files, corresponding to the main elements of the code;
1. Preprocessing - converting Slide-tags output format to work within the structure, and performing other preprocessing or (optional) filtering
2. Reconstruction - The reconstruction requires certain parameters and choices taht is governed by the config file. it is also used to find files from the preprocessing step
3. Analysis - This has several different modes of analysis, and involves going back and forth between functions to a certain degree and uses the config to find the right files
All config files are loaded with default settings that correspond to analyzing the Human tonsil

While most of the code operates on relatively standard python packages (numpy, pandas, matplotlib etc.) packages related to the STRND reconstruction (available here https://github.com/DavidFernandezBonet/Network_Spatial_Coherence) and the reconstructions "morphing" (available here: https://github.com/DavidFernandezBonet/alphamorph) has to be installed. 
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

