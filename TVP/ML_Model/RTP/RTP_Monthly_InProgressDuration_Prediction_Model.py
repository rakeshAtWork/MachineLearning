# Importing necessary libraries and modules
import os   # Importing the os module for operating system related functions
import numpy as np          # NumPy for numerical operations
import pandas as pd         # Pandas for data manipulation and analysis
import datetime             # Datetime module for fetching current time
import shutil               # File processing utility module
from datetime import date   # Date module for fetching current date
from xgboost import XGBRegressor as XGBR # XGB Regressor Model Class
from sklearn.model_selection import (
    GridSearchCV  # GridSearchCV for hyperparameter tuning
    ,train_test_split)  # train_test_split for splitting data
import matplotlib.pyplot as plt   # Matplotlib for plotting
from xgboost import plot_importance  # For plotting the respective importance of individual features
from sklearn.metrics import mean_absolute_percentage_error 
import datetime
import pytz
import businesstimedelta
import holidays as pyholidays # Function for calculating mean absolute percentage error

import logging  



def fill_missing_week(df2):
    unique_weekday = sorted(df2['week no'].value_counts().index.tolist())
    for i in range(1,53):
        unique_weekday = sorted(df2['week no'].value_counts().index.tolist())
        if i not in unique_weekday:
            if i < 2:
                continue
            new_dict = df2[df2['week no'] == i-1].tail(1).to_dict(orient='records')
            index_to_insert = len(df2) 
            print(index_to_insert)
            new_dict[0]['week no'] = i
            new_dict[0]['Final In Progress Duration'] = 0.00
            new_record = new_dict[0]
            print(new_record)
            df2.loc[-1] = new_record 
            df2 = df2.reset_index(drop=True)
    return df2


#function to fill the priority as per the requirement
def priority_conv(val):
  if (val == 1) or (val == 1.0) or (val == '1'):
    return 'P1'
  elif (val == 2) or (val == 2.0) or (val == '2'):
    return 'P2'
  elif (val == 3) or (val == 3.0) or (val == '3'):
    return 'P3'
  elif (val == 4) or (val == 4.0) or (val == '4'):
    return 'P4'
  else:
    return 'P6'
  
#function to convert the time in HH:MM:SS format into hours
def hour_conv(value):
    val_list=value.split(":")
    if len(val_list)==3:
        return np.round(int(val_list[0])+(int(val_list[1])/60) + (int(val_list[2])/3600),2)
    elif len(val_list)==2:
        return np.round(int(val_list[0])+(int(val_list[1])/60),2)
 
def datetime_to_time(value):
    """
    Converts a datetime.datetime object to a time string in the format "hours:minutes:seconds".
    If the input is not a datetime.datetime object, it assumes it's already a datetime.time object and returns it.
   
    Parameters:
    value: A datetime.datetime or datetime.time object.
   
    Returns:
    str: A string representation of time in the format "hours:minutes:seconds".
    """
    value=str(value)
    if ' ' in value:
        date_and_time=value.split(' ')
        year_val,month_val,day_val=date_and_time[0].split('-')
        time_part=date_and_time[1].split(':')
        if len(time_part)==3:
            hour_val=time_part[0]
            minute_val=time_part[1]
            second_val=time_part[2]
        else:
            hour_val=time_part[0]
            minute_val=time_part[1]
            second_val='00'
        value=datetime.datetime(year=int(year_val),month=int(month_val),day=int(day_val),hour=int(hour_val),minute=int(minute_val),second=int(second_val))
    else:
        # try:
        # print(value)
        time_part=value.split(':')
        if len(time_part)==3:
            hour_val=time_part[0]
            minute_val=time_part[1]
            second_val=time_part[2]
        else:
            hour_val=time_part[0]
            minute_val=time_part[1]
            second_val='00'
        value=datetime.time(hour=int(hour_val),minute=int(minute_val),second=int(second_val))

    # Check if the input is a datetime.datetime object
    if isinstance(value, datetime.datetime):
        # Convert years and months into hours (approximation, assuming 30 days per month)
        total_hours = ((value.month-1) * 30 * 24) + ((value.day) * 24) + value.hour
        # Construct and return the time string
        return f"{total_hours}:{value.minute}:{value.second}"
    else:
        # If it's not a datetime.datetime object, assume it's already a datetime.time object and return it
        return f"{value.hour}:{value.minute}:{value.second}"

#function to find the actual 'In progress' duration by removing the outliers
def find_final_rt(actual,avg):
    if avg == 'nan' or avg == np.nan:
        avg = 0
    return min(avg,actual)

#function to find the average 'In progress' duration for each combination of  priority,category,subcategory and ticket type
def find_mean(ipdp_df,priority,category,subcategory,ticket_type):
    return ipdp_df[(ipdp_df['Priority'] == priority) & (ipdp_df['Category'] == category) & (ipdp_df['Ticket Type'] == ticket_type) & (ipdp_df['Subcategory'] == subcategory)]\
["'In progress' duration"].mean()

 

def data_preprocessing(dataset_required, return_whole_encoded_dataset=False):
    """
    Selecting only required columns and renaming them as required
    """
    dataset_required=dataset_required[['Category', 'Subcategory',   'Priority', 'Ticket Type', 
                                       'Call Date','\'In progress\' duration']]
    # dataset_required.rename(columns={'category': 'Category',
    #                             'subcategory': 'Subcategory',
    #                             'priority': 'Priority',
    #                             'ticket_type': 'Ticket Type',
    #                             'number_of_days_current': 'Number of days current'},inplace=True)

    """
    Type Safteying
    """
    dataset_required.replace('',np.nan,inplace=True)
    dataset_required.replace(' ',np.nan,inplace=True)
    dataset_required.replace('nan',np.nan,inplace=True)
    """
    NULL VALUE REMOVAL
    """
    # Remove the all null values from the data
    dataset_required.dropna(inplace=True)


    """
    extract the year,month,day and hour from the call date
    """
    dataset_required['year'] = dataset_required['Call Date'].dt.year
    dataset_required['month'] = dataset_required['Call Date'].dt.month



    """
    APPLY CUSTOM FUNCTIONS TO GET THE DATA IN THE DESIRED FORMAT
    """
    dataset_required['Priority'] = dataset_required['Priority'].apply(priority_conv)
    dataset_required['\'In progress\' duration'] = dataset_required['\'In progress\' duration'].apply(datetime_to_time).apply(hour_conv)
    dataset_required['Average IPD'] =  dataset_required.apply(lambda row: find_mean(dataset_required,row['Priority'],row['Category'],row['Subcategory'],row['Ticket Type']),axis=1)
    dataset_required['Final In Progress Duration'] = dataset_required.apply(lambda row : find_final_rt(row["'In progress' duration"],row['Average IPD']),axis=1)
    df2 = dataset_required.copy()
    # df2 = fill_missing_week(df2)
    dataset_required = df2.copy()
    # """
    # OUTLIER REMOVAL ONLY FROM TRAINING DATA
    # """
    # dataset_required[''In progress' duration after removing outlier'] = dataset_required.apply(lambda row: find_mean(dataset_required,row['Priority'],row['Category'],row['Subcategory'],row['Ticket Type']),axis=1)
    # dataset_required['Final 'In progress' duration'] = dataset_required.apply(lambda row : find_final_rt(row["'In progress' duration"],row[''In progress' duration after removing outlier']),axis=1)
    # dataset_required = dataset_required[dataset_required['Final 'In progress' duration'] < 350]

    """
    REMOVING THE COLUMNS WHICH ARE NOT REQUIRED FOR THE PREDICTION
    """
    dataset_required.drop(columns=['Call Date',"'In progress' duration",'Average IPD'],inplace=True)
    """
    ONE-HOT ENCODING OF CATEGORICAL VARIABLES
    """
    # Convert categorical variables into dummy variables
    # Priority, Category, Subcategory, and Ticket Type columns will be one-hot encoded
    progress_modified = pd.get_dummies(dataset_required, columns=['Priority', 'Category', 'Subcategory', 'Ticket Type','month'])


    """
    FEATURE(INPUT)-LABEL(OUTPUT) SEPARATION
    """
    # Set the output of label column
    label_column= 'Final In Progress Duration'
    
    
    # Drop the target column from the DataFrame and assign the rest to X
    X = progress_modified.drop(label_column, axis=1)
    
    # Assign the target column to y
    y = progress_modified[label_column]

    if not return_whole_encoded_dataset:
        return X, y
    else:
        return X, y, progress_modified 

def train_model(X_train,
                y_train,

                param_grid={                    # Define a dictionary with hyperparameters to be tuned and their respective values
                    "max_depth": [3, 4, 5, 6, 7],
                    "learning_rate": [0.015, 0.02, 0.025, 0.03],
                    "max_leaves": [5, 7, 9, 11, 13],
                    "n_estimators": [200, 300, 400, 500],},
               random_state=42         # Define random state
               ):
    

    
    # Print the shapes of the training and testing sets
    print("Shapes of Training set - Features:", X_train.shape, "Labels:", y_train.shape)
    # print("Shapes of Testing set - Features:", X_test.shape, "Labels:", y_test.shape)

    # # Load model created today or create a new one
    # # if :
    #     # Load existing models created earlier today from a JSON file
    #     regressor=XGBR()
    #     regressor.load_model(f"ipdp_model.json")
    # else:
    #     # Creating an instance of XGBoost Regressor Model class
    regressor=XGBR()
    
    # Initialize GridSearchCV with the regressor (e.g., XGBoost) and the parameter grid
    # cv=5 specifies 5-fold cross-validation
    search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)
    
    # Fit the model with every combination of the above values
    # search.best_params_ returns the best combination of hyperparameters found during the search
    print("The best hyperparameters are ", search.best_params_)

    # Creating an XGBoost regressor with the best parameters obtained from grid search
    regressor = XGBR(learning_rate=search.best_params_["learning_rate"],
                                n_estimators=search.best_params_["n_estimators"],
                                max_leaves=search.best_params_["max_leaves"],
                                max_depth=search.best_params_["max_depth"],
                                eval_metric='rmse')

    
        
    # Fitting the regressor to the training data
    regressor.fit(X_train, y_train)

    return regressor


def ipdp_main(dataset_latest_ticket,model_dir_path='.',model_file_name='IPDP_model.json',test_size=0.3,random_state=23,\
    table_name='InProgressDurationPredictionOutput',\
    logfilename=f"ipdp{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",test=False):

    # Create and configure logger
    logging.basicConfig(filename=logfilename,\
					format='%(asctime)s %(message)s',\
					filemode='w',level=logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)  # Set level to INFO for this handler


    # Creating an object
    logger = logging.getLogger(__name__)

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    logger.info("Logging Started!")

    X,y=data_preprocessing(dataset_latest_ticket)
    logger.info("Data was preprocessed.")

    print(X.head())
    X=X.astype('int')
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1-test_size, test_size=test_size, random_state=random_state)
    logger.info("Test train split was carried out.")
    
    if not test:
        print('##'*23)
        regressor = train_model(
            X_train, y_train,      # Training Data
            # param_grid=,         # Parameter Grid
        random_state=42         # Define random state
        )
    else:
        print('-'*22)
        regressor=XGBR()
        regressor.fit(X_train,y_train)
    logger.info("Model was trained.")

    # Saving the model
    regressor.save_model(model_file_name)
    shutil.move(model_file_name, os.path.join(model_dir_path, model_file_name))  # move model JSON file

    logger.info(f"Model was saved in {model_dir_path} directory.")

import sys
if __name__=='__main__':
    ipdp_main(pd.read_excel(sys.argv[1]), model_dir_path='.')
