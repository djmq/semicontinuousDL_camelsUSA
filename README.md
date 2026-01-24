# semicontinuousDL_camelsUSA

semicontinuousDL_camelsUSA is a set of MATLAB scripts and functions for training the probabilistic deep learning models for semi-continuous hydrological data as described in Quilty and Jahangir (2026): https://doi.org/10.1016/j.jhydrol.2026.134986.

## Installation

1. Download and unzip package in MATLAB working directory. 
2. Right-click package and select -> Add to Path -> Selected Folders and Subfolders
3. Go to data folder, open zenodo_data_download.m and update 'destinationFolder', ensuring the path ends in \data\.
4. Run zenodo_data_download.m
5. Update all directory paths in scripts:
	- deterministic_training_pipeline.m
	- hurdle_training_pipeline.m
	- rg_training_pipeline.m

Note: The code has been tested in MATLAB R2024a. There is no guarantee it will work as intended in previous versions.

## Usage

After following the installation steps, run any of the scripts in step 5 (Installation) to train, validate, and test the related model. 

Note: training time will vary with hardware, so adjust the number of training epochs accordingly. The hurdle model is the most efficient in terms of required training epochs until convergence (usually, between 5-10 epochs).



## License

[MIT](https://choosealicense.com/licenses/mit/)