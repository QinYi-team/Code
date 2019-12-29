1. seven files are offered as follows:
(1) activation_functions_utils: the programs of various activation functions 
(2) data_utils: some program files for dataset such as normalize and binarize
(3) datasets: the planetgear fault data and a excel file used to save and analyze the result
(4) net_utils: the net are built in this file
(5) results: the results after training will be saved in this file
(7) Democomp.m: the main program is used for the comparison of different activation functions.
(8) nnmain.m: this file can run indepdently if necessary parameters are provided.

Notice: run Democomp.m.

2. Introduction of sample set
(1) The test rig
    Drivetrain diagnostics simulator (DDS) is designed by SpectraQuest Inc (the company website can be visited with ¡°http://www.pinxuntech.com/¡±), 
and it mainly consists of a driving motor, a two-stage planetary gearbox, a two-stage paral-lel-axis gearbox, a programmable magnetic brake.

(2) Sample collection:
    We focus on the secondary sun gear of the planetary gearbox because of its higher failure rate than other components in the gearbox. 
Four most typical gear faults including surface wear, crack tooth, chipped tooth and missing tooth are discussed. Meanwhile, a normal gear is taken for comparison.
Vibration signals are collected under four different load conditions (0 Nm, 1.4 Nm, 2.8 Nm and 25.2 Nm respectively)

(3) Feature extraction:
    Impulsive signals are extracted from original ones by Morlet wavelet transform.
    Then both original and impulsive signals are taken to extract features, and totally 50 features are extracted.
    The samples in gear_fault_featuredata_5120.mat is constructed by the above 50 features.