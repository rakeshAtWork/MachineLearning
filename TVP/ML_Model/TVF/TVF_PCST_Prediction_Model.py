# Importing necessary libraries and modules
import os  # Operating System module for interacting with the operating system
import shutil  # File operation module
import zipfile  # Zip module for archieving
import subprocess  # Subprocess module for spawning new processes
import numpy as np  # NumPy for numerical operations
import itertools  # for iterations of functions over an array-like object
import pandas as pd  # Pandas for data manipulation and analysis
from prophet import Prophet  # Prophet library for time series forecasting
from urllib.parse import (
    quote_plus,
)  # URL Parser for concatenating passwords with special characters
from prophet.make_holidays import (
    make_holidays_df,
)  # function for creating a dataframe with local holidays for Prophet
from sklearn.metrics import (
    mean_absolute_percentage_error,  # Mean Absolute Percentage Error (MAPE) metric
    r2_score,  # R-squared score metric
)
from prophet.diagnostics import performance_metrics, cross_validation
from prophet.serialize import (
    model_to_json,
    model_from_json,
)  # Serialization functions for Prophet model
import logging  # importing the logging module
import datetime
from itertools import combinations # importing combinations module
from multiprocessing import Pool  #Pooling class from multiprocessing library
import warnings
# from user_panel.models import (
#     UserPanel,
# )  # datetime module for fetching current timestamp

def date_filler_base(df, date_col, start_date=None, end_date=None):
    if start_date is None:
        start_date = df[date_col].min()
    if end_date is None:
        end_date = df[date_col].max()
    date_range = pd.date_range(start=start_date, end=end_date)

    # Create a new DataFrame with all dates and fill missing values with 0
    new_df = df.set_index(date_col).reindex(date_range).fillna(0)

    # Reset the index to obtain the final DataFrame
    new_df = new_df.reset_index()
    new_df.columns = [date_col] + list(new_df.columns[1:])
    return new_df

def val_encode(element,all_unique_values):
    encoded_dictionary={}
    for i in all_unique_values:
        if i in element:
            encoded_dictionary[i]=1
        else:
            encoded_dictionary[i]=0
    return encoded_dictionary

def save_model(model_name, file_name):
    with open(file_name, "w") as fout:
        fout.write(model_to_json(model_name))  # Save model

def train_regular(param_grid, dataset, best_params=None):
    if best_params is not None:
        model=Prophet(**best_params).fit(dataset)
    else:
        model=Prophet().fit(dataset)
    return model
    
def train_valid_best(param_grid, dataset):
    """
    Trains and evaluates Prophet models with different hyperparameter combinations using cross-validation.

    Args:
        param_grid (dict): A dictionary of hyperparameters and their potential values to explore.
        dataset (pandas.DataFrame): The dataset containing historical time series data for training and validation.

    Returns:
        Prophet: The best-performing Prophet model based on cross-validation results.
    """

    # Generate all combinations of parameters
    all_params = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    # Store RMSEs for each parameter combination
    rmses = []

    # Use cross-validation to evaluate all models
    for params in all_params:
        # Fit model with given parameters
        model = Prophet(**params).fit(dataset)

        # Perform cross-validation with specified settings
        df_cv = cross_validation(
            model,
            horizon="90 days",
            period="30 days",
            initial="30 days",
            parallel="processes",
        )

        # Calculate performance metrics
        df_p = performance_metrics(df_cv, metrics=["rmse"], rolling_window=1)
        rmse = df_p["rmse"].values[0]  # Extract RMSE
        rmses.append(rmse)

    # Find the best parameters based on RMSE scores
    tuning_results = pd.DataFrame(all_params)
    tuning_results["rmse"] = rmses
    #print(tuning_results)  # Display tuning results
    best_params = all_params[np.argmin(rmses)]

    # Fit the final model with the best parameters
    best_model = Prophet(**best_params).fit(dataset)
    return best_model

def load_model(model_path):
    # Open the specified file ('model_path') in read mode
    with open(model_path, "r") as fin:
        # Read the contents of the file into a string ('model_json')
        model_json = fin.read()

    # Deserialize the model from the JSON representation
    model = model_from_json(model_json)

    # Return the deserialized model
    return model

def generate_combinations(list_of_cols):
    # Calculate the length of the input list
    n = len(list_of_cols)
    
    # Initialize an empty list to store combinations
    result = []
    
    # Iterate over the range of 1 to n (inclusive)
    for i in range(1, n + 1):
        # Generate combinations of size i using indexes of list_of_cols
        combinations_of_i = list(combinations(range(n), i))
        
        # Extend the result list with newly generated combinations
        result.extend(combinations_of_i)
    
    # Initialize an empty list to store converted combinations
    # converted_combinations = [()]
    converted_combinations = []

    # Iterate over each combination in the result list
    for combination in result:
        # Convert the combination of indexes to a tuple of corresponding column names
        converted_combinations.append(tuple(list_of_cols[i] for i in combination))
    
    # Return the list of converted combinations
    return converted_combinations

def generate_varname_dataframe(anyiterable):
    anyiterable=[i.lower().replace(' ','_') for i in anyiterable]
    wise_string='_wise_'.join(anyiterable)+'_wise'
    return 'dataset_pivot_'+wise_string+'_daily_count'

def assign_variables(list_varname, values):
    # Create an empty dictionary to store variables and their values
    variables = {}

    # Iterate through the list of variable names and corresponding values
    for varname, value in zip(list_varname, values):
        # Assign the value to the variable name in the dictionary
        variables[varname] = value
    
    # Return the dictionary containing the variables and their values
    return variables

def date_filler(dataset, date_col,first_date=None, last_date=None):
    # Create a range of dates
    dataset[date_col] = pd.to_datetime(dataset[date_col])
    dataset.dropna(inplace=True)

    if first_date is None:
        date_range = pd.date_range(
            start=dataset[date_col].min(), end=dataset[date_col].max()
        )
    else:
        #print(first_date,last_date)
        date_range = pd.date_range(
            start=first_date, end=last_date
        )        
    # Create a new DataFrame with all dates and fill missing values with 0
    new_dataset = dataset.set_index(date_col).reindex(date_range).fillna(0)

    # Reset the index to obtain the final DataFrame
    new_dataset = new_dataset.reset_index()
    new_dataset.columns = [date_col] + list(new_dataset.columns[1:])
    return new_dataset

def data_preprocessing(dataset_latest_ticket, cols_required_list=['Priority','Category','Subcategory','Ticket Type'], return_whole_encoded_dataset=False):
    # Remove rows with empty values
    dataset_latest_ticket.replace('',np.nan,inplace=True)
    dataset_latest_ticket.replace(' ',np.nan,inplace=True)
    dataset_latest_ticket.replace('nan',np.nan,inplace=True)

    dataset_latest_ticket = dataset_latest_ticket.dropna(subset=cols_required_list+['Call Date'])
    dataset_latest_ticket.replace({'1': 'P1', '2': 'P2', '3': 'P3', '4': 'P4'}, inplace=True)

    # Convert 'Call Date' column to datetime and extract date
    dataset_latest_ticket["Call Date"] = pd.to_datetime(dataset_latest_ticket["Call Date"]).dt.date

    # Group by date and count occurrences
    dataset_pivot_daily_count = dataset_latest_ticket.groupby("Call Date")["Ticket Number"].count().reset_index(name="all").fillna(0)

    # Generate all possible combinations of columns required
    all_possible_combinations = generate_combinations(cols_required_list)

    # Initialize lists to store dataframe variable names and dataframes
    dateframe_varname_list = []
    dataframe_list = []

    # Loop through each combination of columns
    for column_or_colcomb in all_possible_combinations:
        # Generate variable name for the current dataframe
        current_dataframe = generate_varname_dataframe(column_or_colcomb)
        dateframe_varname_list.append(current_dataframe)

        # Pivot the dataset and fill NaN values with 0
        pivot_table = dataset_latest_ticket.pivot_table(
            columns=column_or_colcomb,
            index="Call Date",
            values="Ticket Number",
            aggfunc="count"\
        ).fillna(0).reset_index()


        # Fill missing dates in the dataframe
        pivot_table = date_filler(pivot_table, "Call Date",\
                                         pd.to_datetime(dataset_pivot_daily_count['Call Date']).min(),\
                                         pd.to_datetime(dataset_pivot_daily_count['Call Date']).max())

        # Append the pivot table to dataframe list
        dataframe_list.append(pivot_table)

    # Rename column in the main dataframe
    dataset_pivot_daily_count.rename(columns={"Ticket Number": "all"}, inplace=True)

    return (\
        dataset_pivot_daily_count,\
        all_possible_combinations,\
        assign_variables(dateframe_varname_list, dataframe_list)
    )



def train_save_scenario_combinations(all_possible_combinations, dateframe_varname_dateframe_dict, model_dir_path):
    for column in all_possible_combinations[len(all_possible_combinations) - 1]:
        var_name = f"{column.lower().replace(' ', '_')}_var"
        exec(f"{var_name}=None")
        #print(var_name)
    for comb_serial_no, comb in enumerate(all_possible_combinations):
        dataset_current_pivotted = dateframe_varname_dateframe_dict[
            list(dateframe_varname_dateframe_dict.keys())[comb_serial_no]
        ]

        for i in dataset_current_pivotted.columns:
            if i not in ["Call Date", "index", "DateYMD"]:
                dataset_current = dataset_current_pivotted[["Call Date", i]].copy()
                dataset_current.columns = ["ds", "y"]
                #print(dataset_current.head())
                dataset_current = date_filler(dataset_current, "ds")
                dataset_current["ds"] = pd.to_datetime(dataset_current["ds"])
                #print(dataset_current.head())
                try:
                    
                    if dataset_current['y'].median()>=5:
                        dataset_current["cap"] = (dataset_current["y"].max() * 2) + 1
                    else:
                        dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
                    
                    #dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
                except:
                    print(dataset_current.head())
                dataset_current["floor"] = 0
                model_current = train_regular(None, dataset_current)

                vars_dict = {}
                for comb_el_cnt, comb_el in enumerate(comb):
                    var_name = f"{comb_el.lower().replace(' ', '_')}_var"
                    if isinstance(i, str):
                        vars_dict[var_name] = i.replace('/', 'SLASH')
                    else:
                        vars_dict[var_name] = i[comb_el_cnt].replace('/', 'SLASH')

                model_filename = f'''category_{vars_dict.get('category_var', 'all')}_priority_{vars_dict.get('priority_var',
'all')}_ticket_type_{vars_dict.get('ticket_type_var', 'all')}_subcategory_{vars_dict.get('subcategory_var', 'all')}_model.json'''
                #print(model_filename)
                save_model(model_current, model_filename)
                #print(i, " Finished!")

def train_save_scenario(comb_serial_no, comb,dataset_current_pivotted, model_dir_path):
    print(comb)
    for i in dataset_current_pivotted.columns:
        if i not in ["Call Date", "index", "DateYMD"]:
            dataset_current = dataset_current_pivotted[[i, "Call Date"]].copy()
            dataset_current.columns = ["y","ds"]
            #dataset_current = date_filler(dataset_current, "ds")
            ##print(dataset_current.head())
            dataset_current["ds"] = pd.to_datetime(dataset_current["ds"])
            
            if dataset_current['y'].median()>=5:
                dataset_current["cap"] = (dataset_current["y"].max() * 2) + 1
            else:
                dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
            
            #dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
            dataset_current["floor"] = 0
            model_current = train_regular(None, dataset_current)

            vars_dict = {}
            for comb_el_cnt, comb_el in enumerate(comb):
                var_name = f"{comb_el.lower().replace(' ', '_')}_var"
                if isinstance(i, str):
                    vars_dict[var_name] = i.replace('/', 'SLASH')
                else:
                    vars_dict[var_name] = i[comb_el_cnt].replace('/', 'SLASH')

            model_filename = f'''category_{vars_dict.get('category_var', 'all')}_priority_{vars_dict.get('priority_var',
'all')}_ticket_type_{vars_dict.get('ticket_type_var', 'all')}_subcategory_{vars_dict.get('subcategory_var', 'all')}_model.json'''
            #print(model_filename)
            if len(comb)==4:
                print(comb)
                print(model_filename)
            save_model(model_current, os.path.join(model_dir_path, model_filename))
            #print(i, " Finished!")

def train_save_scenario_combinations_parallel(all_possible_combinations, dataframe_varname_dateframe_dict, model_dir_path):
    model_dir_path_list=[model_dir_path for i in range(len(all_possible_combinations))]
    with Pool() as pool:
        pool.starmap(train_save_scenario, [(comb_serial_no, comb,list(dataframe_varname_dateframe_dict.values())[comb_serial_no],\
         model_dir_path_list[comb_serial_no]) for comb_serial_no, comb in enumerate(all_possible_combinations)])

def train_save_scenario(comb_serial_no, comb,dataset_current_pivotted, model_dir_path):
    for i in dataset_current_pivotted.columns:
        if i not in ["Call Date", "index", "DateYMD"]:
            dataset_current = dataset_current_pivotted[[i, "Call Date"]].copy()
            dataset_current.columns = ["y","ds"]
            #dataset_current = date_filler(dataset_current, "ds")
            print(dataset_current.head())
            dataset_current["ds"] = pd.to_datetime(dataset_current["ds"])
            
            if dataset_current['y'].median()>=5:
                dataset_current["cap"] = (dataset_current["y"].max() * 2) + 1
            else:
                dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
            
            #dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
            dataset_current["floor"] = 0
            model_current = train_regular(None, dataset_current)

            vars_dict = {}
            for comb_el_cnt, comb_el in enumerate(comb):
                var_name = f"{comb_el.lower().replace(' ', '_')}_var"
                if isinstance(i, str):
                    vars_dict[var_name] = i.replace('/', 'SLASH')
                else:
                    vars_dict[var_name] = i[comb_el_cnt].replace('/', 'SLASH')

            model_filename = f'''category_{vars_dict.get('category_var', 'all')}_priority_{vars_dict.get('priority_var',
'all')}_ticket_type_{vars_dict.get('ticket_type_var', 'all')}_subcategory_{vars_dict.get('subcategory_var', 'all')}_model.json'''
            print(model_filename)
            save_model(model_current, os.path.join(model_dir_path, model_filename))
            print(i, " Finished!")

def train_save_scenario_combinations_parallel(all_possible_combinations, dataframe_varname_dateframe_dict, model_dir_path):
    model_dir_path_list=[model_dir_path for i in range(len(all_possible_combinations))]
    with Pool() as pool:
        pool.starmap(train_save_scenario, [(comb_serial_no, comb,list(dataframe_varname_dateframe_dict.values())[comb_serial_no],\
         model_dir_path_list[comb_serial_no]) for comb_serial_no, comb in enumerate(all_possible_combinations)])

def train_save_models(
    dataset_pivot_daily_count,
    all_possible_combinations,
    dateframe_varname_dateframe_dict,
    model_dir_path=".",
    zip_file_name="model_archive.zip",
    holidays=None,
    random_state=42,
    begin_from=0,
    parellelize=True
):
    if holidays is None:
        holidays = make_holidays_df(year_list=[2023 + i for i in range(10)], country="DNK")

    param_grid = {
        "holidays": [holidays],
        "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
        "seasonality_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "holidays_prior_scale": [0.01, 0.1, 1.0, 10.0],
        "seasonality_mode": ["additive", "multiplicative"],
    }

    for i in dataset_pivot_daily_count.columns:
        if i not in ["Call Date", "index", "DateYMD"]:
            dataset_current = dataset_pivot_daily_count[["Call Date", i]].copy()
            dataset_current.columns = ["ds", "y"]
            dataset_current["ds"] = pd.to_datetime(dataset_current["ds"])
            
            if dataset_current['y'].median()>=5:
                dataset_current["cap"] = (dataset_current["y"].max() * 2) + 1
            else:
                dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
            
            #dataset_current["cap"] = (dataset_current["y"].max() * 1.1) + 1
            dataset_current["floor"] = 0
            model_current = train_regular(param_grid, dataset_current)
            save_model(
                model_current,
                os.path.join(model_dir_path, f"category_all_priority_all_ticket_type_all_subcategory_all_model.json"),
            )
            #print(i, " Finished!")
    if parellelize:
        train_save_scenario_combinations_parallel(all_possible_combinations, dateframe_varname_dateframe_dict, model_dir_path)
    else:
        train_save_scenario_combinations(all_possible_combinations, dateframe_varname_dateframe_dict, model_dir_path)
    

def tvf_main(
    dataset_latest_ticket,
    model_dir_path="../media/tvp_models",
    first_date=None,
    end_date=None,
    predict_output=False,
    load_into_db=False,
    table_name="TicketVolumeForecastOutput",
    logfilename=f"tvf{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    begin_from=0,
    parellelize=True
):
    warnings.filterwarnings('ignore')

    # Create and configure logger
    logging.basicConfig(
        filename=logfilename,
        format="%(asctime)s %(message)s",
        filemode="w",
        level=logging.INFO,
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Set level to INFO for this handler

    # Creating an object
    logger = logging.getLogger(__name__)

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    logger.info("Logging Started!")

    (
        dataset_pivot_daily_count,
        all_possible_combinations,
        dateframe_varname_dateframe_dict
    ) = data_preprocessing(dataset_latest_ticket)
    logger.info("Data was preprocessed.")

    train_save_models(
        dataset_pivot_daily_count,
        all_possible_combinations,
        dateframe_varname_dateframe_dict,
        # test_size=0.2,         # Define the test size
        # param_grid=,         # Parameter Grid
        random_state=42,  # Define random state
        model_dir_path=model_dir_path,  # model saving directory path
        begin_from=begin_from,
        parellelize=parellelize
    )

    logger.info("Model was trained.")
    logger.info(f"Model was saved in {model_dir_path} directory.")

import sys
if __name__=='__main__':
    dataset_latest_ticket=pd.read_excel(sys.argv[1])
    tvf_main(dataset_latest_ticket, model_dir_path=sys.argv[2],parellelize=eval(sys.argv[3]))
