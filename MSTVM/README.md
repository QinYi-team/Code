# MSTVM (  'DDC_MSTVM'and'RevGrad_MSTVM')

## Requirement
* python 3
* pytorch(GPU) 1.2

## Usage

#Input data format
1. N *3072 as an array and saved as a ".npz "file.
2. The file contains 'sample', 'label' and 'load' variables.Where N *3072 array data are stored in 'sample',
 'label' stores five categories of corresponding labels of '0 ,1,2,3, 4', and 'load' is the corresponding condition
(this variable can be omitted, and the program in the file "data_loader_utils" can be modified accordingly).


Before running the main program, you need to pre-train to get a pre-training model.
1.Run the file "pretrained_models_rand2/ 程序包 /main_ch".
2.Move the obtained ".pt "file to the directory" pretrained_models_rand2"

You can then run the file "main_ch1331" in your home directory.
(Note that some filenames and addresses are changed.)




