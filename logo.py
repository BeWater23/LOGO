#!/usr/bin/env python3
import sys
import os
import re
import copy
import random
import itertools
import statistics
import multiprocessing
import time
import contextlib

# data wrangling
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# custom
import mlr_utils

# --------------------------------------------------------------------------------------------------------------
# Set up the Modeling Parameters
# --------------------------------------------------------------------------------------------------------------

###-------- Choose substrate or catalyst LOGO ------------###
which_logo = 'substrate'
###-------- Choose model parameters ----------###
n_steps = 2 # This is the maximum number of parameters you want in your models
n_candidates = 50 # This is a measure related to how many models are considered at each step. See mlr_utils.bidirectional_stepwise_regression for more details.
collinearity_cutoff = 0.6 # This is collinearity (r^2) above which parameters won't be included in the same model

# --------------------------------------------------------------------------------------------------------------
# Reading in the data 
# --------------------------------------------------------------------------------------------------------------
# This cell assumes that your spreadsheets are in .xlsx format and that there are no columns after the parameters
# or rows after the last reaction. Extra rows and columns before the data are fine and can be skipped with the
# parameters_header_rows and parameters_start_col variables.
# Check cell outputs to make sure everything looks good

parameters_sheet = "Sheet1" # Sheet in the Excel file to pull parameters from
parameters_start_col = 2   # 0-indexed column number where the parameters start
parameters_y_label_col = 0  # 0-indexed column number where the ligand labels are
parameters_header_rows = 0 # Number of rows to skip when reading the parameters
response_sheet = "Sheet1" # Sheet in the Excel file to pull responses from
response_col = 1 # 0-indexed column number for the responses
response_y_label_col = 0  # 0-indexed column number where the ligand labels are
response_header_rows = 0 # Number of rows to skip when reading the responses

RESPONSE_LABEL = "ddG" # Name of your response variable

# --------------------------------------------------------------------------------------------------------------
# EDIT ABOVE THIS LINE
# --------------------------------------------------------------------------------------------------------------

# Set the number of processors to use for parallel processing
# Respect SLURM allocation if available, otherwise fall back to local cpu_count()-2
total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count() - 2))

# Always at least 2 cores for the inner stepwise regression
inner_threads = max(2, total_cores // 8)  # heuristic: at least 1/8 of cores per inner job

# Then decide how many LOGO workers we can run at once
n_logo_workers = max(1, total_cores // inner_threads)

print(f"LOGO parallelization plan: {n_logo_workers} workers × {inner_threads} threads each (total {n_logo_workers*inner_threads}/{total_cores} cores)")
print("LOGO Type:", which_logo)

# Get spreadsheet filename from command line
if len(sys.argv) < 2:
    print("Usage: python script.py <spreadsheet.xlsx>")
    sys.exit(1)

input_file = sys.argv[1]  # first argument after script name

# Remove the extension and add "_logo"
base_name = os.path.splitext(os.path.basename(input_file))[0]  
output_folder = base_name + "_" + which_logo + "_logo"                           

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Actually start reading stuff into dataframes
parameters_df = pd.read_excel(input_file,
                              parameters_sheet,
                              header = parameters_header_rows,
                              index_col = parameters_y_label_col,
                              )
response_df = pd.read_excel(input_file,
                            response_sheet,
                            header = response_header_rows,
                            index_col = response_y_label_col,
                            usecols = list(range(0, response_col + 1))
                            )


# Drop any columns before parameters_start_col that are not the index column
parameters_columns_to_keep = [col for col in range(0, len(parameters_df.columns)) if col >= parameters_start_col-1]
parameters_df = parameters_df.iloc[:,parameters_columns_to_keep]

# Combine the two dataframes into the master dataframe
response_df.drop(response_df.columns[0:response_col-1], axis = 'columns', inplace = True)
data_df = response_df.merge(parameters_df, left_index = True, right_index = True)
data_df.rename(columns = {data_df.columns.values[0]: RESPONSE_LABEL}, inplace = True) # Converts the output column name from whatever it is on the spreadsheet
data_df.dropna(inplace = True) # Remove any rows with blanks

# This converts all the data to numeric values since it was reading them in as non-numeric objects for some reason
for column in data_df.columns:
    data_df[column] = pd.to_numeric(data_df[column], errors='coerce')

# Get a list of all the features
all_features = list(data_df.columns)
all_features.remove(RESPONSE_LABEL)

# Check for duplicate reaction labels or column names
error = False
if len(list(data_df.index)) != len(list(set(data_df.index))):
    print('THERE ARE DUPLICATE REACTION LABELS IN THE DATA. PLEASE CORRECT THIS IN YOUR SPREADSHEET.')
    error = True
if len(list(data_df.columns)) != len(list(set(data_df.columns))):
    print('THERE ARE DUPLICATE COLUMN NAMES IN THE DATA. PLEASE CORRECT THIS IN YOUR SPREADSHEET.')
    error = True

if not error:
    # Print out the data distribution of the response variable
    plt.figure(figsize=(5, 5))
    plt.hist(data_df[RESPONSE_LABEL], color='grey')
    plt.xlabel(RESPONSE_LABEL, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.savefig(f"{output_folder}/histogram.png", dpi=300, bbox_inches="tight")  # save instead of show
    plt.close()

#----------------------------------------------------------------------------------------------------------------
# LOGO Modeling
#----------------------------------------------------------------------------------------------------------------

logo = LeaveOneGroupOut()
# Split the cat_substrate in the input file
logo_df = data_df
new_cols = logo_df.index.to_frame(name="cat_substrate")  # index to DataFrame
new_cols[["catalyst", "substrate"]] = new_cols["cat_substrate"].str.split("_", expand=True)
logo_df = pd.concat([logo_df, new_cols], axis=1)
groups = logo_df[which_logo].values
groups = logo_df[which_logo].values
print(f"Groups ({len(np.unique(groups))} total): {np.unique(groups)}\n")
#print(groups)

#------------------------------------------LOGO FUNCTION FOR INDIVIDUAL MODEL TRAINING---------------------------------------------------------------#
##Define the individual MLR training and testing: Similar to the normal Mattlab workflow##
def logo_mlr(X_train, X_test, y_train, y_test, test_group, n_steps, n_candidates, collinearity_cutoff, inner_threads):

    #Create a .log file for each worker to get readable output
    log_file = f"{output_folder}/LOGO_{test_group}.log"
    
    #Open the log file to write all output there
    with open(log_file, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"Left out: {test_group}\n")

        #Scaling of the parameters
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index) # fit on training 
        X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)     # apply to test
        #print(X_test_scaled)

        #Creating a dataframe that is needed for the bidirectional stepwise MLR as input
        train_df = pd.concat([X_train_scaled, y_train], axis=1)
        test_df  = pd.concat([X_test_scaled,  y_test], axis=1)
        #print(train_df)
        #print(test_df)

        #Run stepwise regression
        results,models,sortedmodels,candidates = mlr_utils.bidirectional_stepwise_regression(train_df,RESPONSE_LABEL,
                        n_steps=n_steps,n_candidates=n_candidates,collinearity_cutoff=collinearity_cutoff, n_processors=inner_threads)
    
        for i in results.index:
            model_terms = results.loc[i,"Model"]
            model = models[model_terms].model
            # Set the train MAE and RMSE for each model
            x_train = train_df.loc[:,model_terms]
            y_train = train_df[RESPONSE_LABEL]
            y_predictions_train = model.predict(x_train)
            results.loc[i, 'MAE'] = metrics.mean_absolute_error(y_train, y_predictions_train)
            results.loc[i, 'RMSE'] = np.sqrt(metrics.mean_squared_error(y_train, y_predictions_train))

        # Identify the best model from the bidirectional_stepwise_regression algorithm
        selected_model_terms = results.loc[0, "Model"] # Store a tuple of 'xIDs' for the best model
        selected_model = models[selected_model_terms].model # Store the LinearRegression object for that model

        # Break up the train/test data into smaller dataframes for easy reference
        x_train = train_df.loc[:,selected_model_terms] # Dataframe containing just the parameters used in this model for the ligands used in the training set
        x_test = test_df.loc[:,selected_model_terms] # Dataframe containing just the parameters used in this model for the ligands used in the test set
        y_train = train_df[RESPONSE_LABEL]
        y_test = test_df[RESPONSE_LABEL]

        # Predict the train and test sets with the model
        y_predictions_train = selected_model.predict(x_train)
        y_predictions_test = selected_model.predict(x_test)

        #Plot the final model
        mlr_utils.plot_MLR_model(y_train, y_predictions_train, y_test, y_predictions_test, output_label=RESPONSE_LABEL, plot_xy=True, save_path=f"{output_folder}/MLR_plot_{test_group}.png")

        #Print equation
        print(f'\nParameters:\n{selected_model.intercept_:10.4f} +')
        for i, parameter in enumerate(selected_model_terms):
            print(f'{selected_model.coef_[i]:10.4f} * {parameter}')
        print("\n")

        #Print predicted vs measured ddG for left out group
        test_identifiers = X_test.index
        print(f"\nPredicted vs Measured for left-out group: {test_group}")
        for name, true_val, pred_val in zip(test_identifiers, y_test, y_predictions_test):
            print(f"{name}: Measured = {true_val:.3f} | Predicted = {pred_val:.3f}")
        print("\n")
        
    
    #Evaluate Model Stats
    logo_results = {
        "Left Out Group": test_group,
        "Training R^2": np.round(selected_model.score(x_train, y_train), 3),
        "Training MAE": np.round(metrics.mean_absolute_error(y_train,y_predictions_train), 3),
        "Training RMSE": np.round(np.sqrt(metrics.mean_squared_error(y_train, y_predictions_train)), 3),
        "LOGO Test R^2": np.round(mlr_utils.get_r2(y_test,y_predictions_test,y_train,'out-of-sample'), 3),
        "LOGO MAE": np.round(metrics.mean_absolute_error(y_test,y_predictions_test), 3),
        "LOGO RMSE": np.round(np.sqrt(metrics.mean_squared_error(y_test,y_predictions_test)), 3),
    }

    # Print a short progress marker to stdout
    print(f"[LOGO] Finished group {test_group}", flush=True)
    
    return logo_results

# ------------------------------- LOGO LOOP -------------------------------------------------------------------------------------------------------------#

start_time = time.perf_counter()  # start timing for the LOGO process
# ------------------------ PREPARE INPUTS ------------------------
logo_inputs = []
#Prepare input list
for train_idx, test_idx in logo.split(parameters_df, response_df, groups):
    # Create train/test subsets
    X_train, X_test = parameters_df.iloc[train_idx], parameters_df.iloc[test_idx]
    y_train, y_test = response_df.iloc[train_idx], response_df.iloc[test_idx]
    #print(X_train)
    #print(X_test)
    test_group = groups[test_idx][0]
    logo_inputs.append((X_train, X_test, y_train, y_test, test_group, n_steps, n_candidates, collinearity_cutoff, inner_threads))

# ------------------------ RUN LOGO IN PARALLEL ------------------------
logo_result_list = Parallel(n_jobs=n_logo_workers)(
    delayed(logo_mlr)(*args) for args in logo_inputs
)
# ------------------------ COLLECT RESULTS ---------------------------
end_time = time.perf_counter()   
elapsed = end_time - start_time
print(f"Total elapsed LOGO time: {elapsed/60:.2f} minutes")
print("\n")
logo_results_df = pd.DataFrame(logo_result_list)
logo_results_df_sorted = logo_results_df.sort_values(by="LOGO MAE", ascending=False)
print("Final LOGO Results:")
print("\n")
print(logo_results_df_sorted)
logo_results_df_sorted.to_csv(f"{output_folder}/LOGO_results.csv", index=False)
