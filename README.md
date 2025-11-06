IMPORTANT: Raw data files used to generate the results seen in the corresponding PVP-IPFM paper can be found on [Dropbox](https://www.dropbox.com/scl/fo/egx7hkhad83lg6gervp3o/APNkwt5TZJsS-P4wQvEtdGE?rlkey=0fnpi931hk9844krbcobjszl8&st=d9brrxda&dl=0). Please download it, rename it 'Data', and place the folder in the main directory of where the code is run.

# PVP-IPFM Logistic Regression Trainer (MATLAB with SQL compatibility)
Study the categorizing properties of Peripheral Venous Pressure (PVP) signals! (a.k.a. waveforms) This project is a MATLAB-based trainer for logistic regression models.

Logistic regression allows users to divide data into halves, and models can be trained to predict the identity of a sample up to very high accuracies. PVP signals are measured from the blood pressure in a patient, so it effectively measures heartbeat and respiration. In the dataset provided, infants were non-invasively measured and labeled as either "resuscitated" (hydrated) or hypovolemic ("dehydrated"). The application of this goes beyond dehydration, however. Recent findings suggest there to be a correlation between "hemorrhage" (bleeding) and PVP data. Collecting accurate data to verify this is ethically difficult, but two pilot animal subjects have already been utilized for current analysis (rest their souls).

Results can be stored either locally in an Excel file or in an SQL database.

## Introduction
To use this code, you must run it in MATLAB 2024b or higher. The parallelization toolbox is used in the current implementation but can be turned "off" in settings. Additionally, the database toolbox and several others are required, both to simulate and to upload simulation results to MySQL. Commands are included in the code to automatially create the needed tables for MySQL, so long as the correct database is selected.

## Instructions
The code included for the main simulator is lengthy and may be confusing so here is an overview of how it works:

1. saved_profiles.m includes the profiles, and the MATLAB app PVP_IPFM.mlapp controls the selection of most settings.
2. Inside sim_head.m file, this is how the settings are distributed to the rest of the sections of the project. Here, the settings are made active, connections to a MySQL network (or Excel table) attempt, a child profile is made from each data point needed to generate the figure, then a double for-loop iteratively simulates the needed models, before finally generating the figure. (Empty data points queried on "Generate Figure" will just return null).
3. model_fun_v3.m is where models of all kinds of parameters are generated. There are many settings to specify in the saved_profiles.m. (NEEDED KNOWLEDGE: logistic regression, principal component analysis, ordinal regression)
4. Be sure to configure where you're saving to! Using the Excel method may be glitchy for small incremental frames value.

Alternatively, MAIN_sample_data.m offers a statistical overview of the dataset being presented to the system, including a KS two-sample test, empirical CDF's, and power spectral density plots.

## Configuration Setup
There is a file named "saved_profiles". There you will see several pre-configured profiles to use as reference for your own profile. Profiles work by defining the primary variable for a parametric sweep and the corresponding range. If a figure is being rendered, this is the range of the plot, and each line of the parameter 'configs' specifies a line on the plot, and each line has its own custom parameters separate from those specified in default_parameters. Once all the configs are defined, the user can be specific in defining the appearance of plots using several customizable parameters.

ROC curves are a measure of the estimator's ability to balance false positives from false negatives, and is reflected in a plot of False Positive Rate vs. True Positive Rate.

## Further Questions
For any questions please contact jrwimer@uark.edu or visit [my website](https://jrw-lab.github.io). 
