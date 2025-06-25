
# Setting up the environment and run a first-pass reconstruction

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

Data included in the repository are either publically available from the Russel et al. areticle, or generated from only publically available data. The bead barcodes of the tonsil sample, which is not publically available, has been replaced with synthetic, random barcodes.
Since the preprocessed edgelists are available in the "intermediary_files", the first first part of raw edge file conversion of the preprocessing is not required to be ran, and that cell can therefore be skipped in the notebook.

Before running the code, the 10X barcodes file "3M-february-2018.txt.gz" in the "Input_files" folder should be decompressed. This is required for the preprocessing, and can be done with for eaxample in bash running

```bash
gunzip Input_files\3M-february-2018.txt.gz
```
After setting up the environment, the inital processing and a first-pass reconstruction can be performed using the "Preprocessing.ipynb" notebook.
Alternatively, the individual scripts can be run individually and the the order to run is:
1. Utils.py
2. initital_processing_functions.py
3. filtering_functions.py
4. reconstruction_functions.py
And for reconstruction analysis then run "subgraph_analysis_functions.py"

This there are multiple options for further analysis:
1. Perform more in-depth analysis on the reconstruction with many options in the "additional_subgraph_analysis" notebook
2. Perform an iterative reconstruction in the "subgraph_modification" notebook
3. Perform the biological analysis using the R-based "slidetags-network" R project

To asess non-reconstruction related network properties, the "Base_network_analysis" notebook allows analysis of degree distributions etc. for multiple samples simultaneously

# Iterative reconstruction
Simply open and run the "_Subgraph_modification" notebook

The "subgraph_modification_functions.py" is not independantly runnable without modifying the code, and it is recommended to use the notebook.

# Biological analysis

Biological analysis is run by .qmd R files and a jupyter notebook from the "analysis" subfolder within the "Spatial_biology_analysis_R_squidpy" folder

The input data is currently empty but is publically available as a part of the the original Slide-tags article: https://doi.org/10.1038/s41586-023-06837-4

Non-sequencing Human tonsil data [SCP2169](https://singlecell.broadinstitute.org/single_cell/study/SCP2169/slide-tags-snrna-seq-on-human-tonsil#/)

Mouse data sets are also publically available including sequencing data.

[SCP2162](https://singlecell.broadinstitute.org/single_cell/study/SCP2162/slide-tags-snrna-seq-on-mouse-hippocampus) (Mouse hippocampus) 

[SCP2170](https://singlecell.broadinstitute.org/single_cell/study/SCP2170/slide-tags-snrna-seq-on-mouse-embryonic-e14-brain) (Mouse embryonic brain)

Mouse sequencing data. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244355

The R environment can be recreated using "renv", running the slidetags-network.Rproj will activate the inital renv, and subsequently executing the command below will install required packages 
```R
renv::restore()
```

Then open and execute the "Slide-tags_HuTonsil_snData_part1.qmd", following which the squidpy notebook "slidetags_tonsil_analysis_assayData.ipynb" should be executed by first creating the conda environment for the .yml in the "Spatial_biology_analysis_R_squidpy" folder and choosing it as the notebook kernel:

```bash 
conda env create -f squidpy_environment.yml
```

After squidpy, in R open and execute the "Slide-tags_HuTonsil_snData_part2.qmd" is the last step


