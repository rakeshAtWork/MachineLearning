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
import sys          # system module for receiving argument from CLI
import businesstimedelta
import holidays as pyholidays # Function for calculating mean absolute percentage error

import logging  

def fill_missing_week(df2):
    unique_weekday = sorted(df2['week_no'].value_counts().index.tolist())
    for i in range(1,53):
        unique_weekday = sorted(df2['week_no'].value_counts().index.tolist())
        if i not in unique_weekday:
            new_dict = df2.iloc[-1].to_dict()
            index_to_insert = len(df2) 
            # print(index_to_insert)
            new_dict['week_no'] = i
            new_dict['Final Resolution Time'] = 0.00
            new_record = new_dict
            # print(new_record)
            df2.loc[-1] = new_record 
            df2 = df2.reset_index(drop=True)
    return df2

def fill_missing_month(df2):
    unique_weekday = sorted(df2['month'].value_counts().index.tolist())
    for i in range(1,13):
        unique_weekday = sorted(df2['month'].value_counts().index.tolist())
        if i not in unique_weekday:
            new_dict = df2.iloc[-1].to_dict()
            index_to_insert = len(df2) 
            print(index_to_insert)
            new_dict['month'] = i
            new_dict['Final Resolution Time'] = 0.00
            new_record = new_dict
            print(new_record)
            df2.loc[-1] = new_record 
            df2 = df2.reset_index(drop=True)
    return df2


def fill_missing_sub(df2,df):
    unique_sub = sorted(df2['Subcategory'].value_counts().index.tolist())
    actual_sub = sorted(df['Subcategory'].value_counts().index.tolist())
    if ' ' in actual_sub:
        actual_sub.remove(' ')
    for i in range(len(actual_sub)):
        unique_sub = sorted(df2['Subcategory'].value_counts().index.tolist())
        if actual_sub[i] not in unique_sub:
            new_dict = df2.iloc[-1].to_dict()
            index_to_insert = len(df2) 
            # print(index_to_insert)
            new_dict['Subcategory'] = actual_sub[i]
            new_dict['Final Resolution Time'] = 0.00
            new_record = new_dict
            # print(new_record)
            df2.loc[-1] = new_record 
            df2 = df2.reset_index(drop=True)
    return df2
 
    
def fill_missing_priority(df2,df):
    unique_sub = sorted(df2['Priority'].value_counts().index.tolist())
    actual_sub = sorted(df['Priority'].value_counts().index.tolist())
    print(unique_sub,actual_sub)
    for i in range(len(actual_sub)):
        unique_sub = sorted(df2['Priority'].value_counts().index.tolist())
        if actual_sub[i] not in unique_sub:
            new_dict = df2.iloc[-1].to_dict()
            index_to_insert = len(df2) 
            # print(index_to_insert)
            new_dict['Priority'] = actual_sub[i]
            new_dict['Final Resolution Time'] = 0.00
            new_record = new_dict
            # print(new_record)
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
  
#function to find the business difference between two dates
def find_work_hour_diff(first_date, last_date, work_hour_start="08:00:00", work_hour_end="16:00:00",\
                        lunch_break_start="14:00:00",lunch_break_end="15:00:00", working_days=[0, 1, 2, 3, 4],holidays=[]):
    """
    Calculates the difference between work hours and actual hours between two dates,
    considering work hours, holidays, and cases where dates coincide or end before work hours.

    Args:
        first_date (str): Start date.
        last_date (str): End date.
        work_hour_start (str, optional): Work hour start time. Defaults to "09:30:00".
        work_hour_end (str, optional): Work hour end time. Defaults to "18:30:00".
        holidays (list, optional): List of holidays. Defaults to [].

    Returns:
        tuple: A tuple containing the following:
            total_business_hours (float): Total number of business hours.
            actual_hours (float): Total number of actual hours.
            number_of_business_days (int): Number of business days.
    """
    work_hour_start_list=[int(x) for x in work_hour_start.split(':')]
    work_hour_end_list=[int(x) for x in work_hour_end.split(':')]
    lunch_break_start_list=[int(x) for x in lunch_break_start.split(':')] 
    lunch_break_end_list=[int(x) for x in lunch_break_end.split(':')]
    # Define a working day
    workday = businesstimedelta.WorkDayRule(
        start_time=datetime.time(work_hour_start_list[0],work_hour_start_list[1]),
        end_time=datetime.time(work_hour_end_list[0],work_hour_end_list[1]),
        working_days=working_days)

    # Take out the lunch break
    lunchbreak = businesstimedelta.LunchTimeRule(
        start_time=datetime.time(lunch_break_start_list[0],lunch_break_start_list[1]),
        end_time=datetime.time(lunch_break_end_list[0],lunch_break_end_list[1]),
        working_days=working_days)

    ca_holidays = pyholidays.DNK()
    holidays = businesstimedelta.HolidayRule(ca_holidays)

    # Combine the two
    # businesshrs = businesstimedelta.Rules([workday, lunchbreak, holidays])
    businesshrs = businesstimedelta.Rules([workday,holidays])

    
    first_date, last_date = pd.to_datetime(first_date), pd.to_datetime(last_date)
    
    start = datetime.datetime(first_date.year, first_date.month,\
                              first_date.day, first_date.hour,\
                              first_date.minute, first_date.second)
    end = datetime.datetime(last_date.year, last_date.month,\
                              last_date.day, last_date.hour,\
                              last_date.minute, last_date.second)
    bdiff = businesshrs.difference(start, end)
    
    return int(bdiff.hours)

#function to find the actual resolution time by removing the outliers
def find_final_rt(actual,avg):
    if avg == 'nan' or avg == np.nan:
        avg = 0
    return min(avg,actual)

#function to find the average resolution time for each combination of  priority,category,subcategory and ticket type
def find_mean(rtp_df,priority,category,subcategory,ticket_type):
    return rtp_df[(rtp_df['Priority'] == priority) & (rtp_df['Category'] == category) & (rtp_df['Ticket Type'] == ticket_type) & (rtp_df['Subcategory'] == subcategory)]\
['Resolution Time'].mean()

 

def data_preprocessing(dataset_required,ticket_type, return_whole_encoded_dataset=False):
    """
    Selecting only required columns and renaming them as required
    """
    df = dataset_required.copy()
    dataset_required=dataset_required[['Category', 'Subcategory',   'Priority', 'Ticket Type', 
                                       'Call Date','Completion Date']]
    # dataset_required.rename(columns={'category': 'Category',
    #                             'subcategory': 'Subcategory',
    #                             'priority': 'Priority',
    #                             'ticket_type': 'Ticket Type',
    #                             'number_of_days_current': 'Number of days current'},inplace=True)

    """
    Type Safteying
    """
    dataset_required['Completion Date'] = dataset_required['Completion Date'].fillna(np.nan)
    dataset_required.replace('',np.nan,inplace=True)
    dataset_required.replace(pd.NaT,np.nan,inplace=True)
    dataset_required.replace(' ',np.nan, inplace=True)
    dataset_required.replace('nan',np.nan,inplace=True)
    """
    NULL VALUE REMOVAL
    """
    dataset_required['Completion Date'].dropna(inplace=True)
    dataset_required['Call Date'].dropna(inplace=True)
    dataset_required.dropna(subset=['Completion Date'],inplace=True)
    dataset_required.dropna(inplace=True)


    """
    extract the year,month,day and hour from the call date
    """
    dataset_required['Call Date']=pd.to_datetime(dataset_required['Call Date'])
    dataset_required['Completion Date']=pd.to_datetime(dataset_required['Completion Date'])
    dataset_required['year'] = dataset_required['Call Date'].dt.year
    dataset_required['month'] = dataset_required['Call Date'].dt.month
    # dataset_required['week_no']=dataset_required['Call Date'].dt.isocalendar().week
    dataset_required = dataset_required[dataset_required['Ticket Type'] == ticket_type]




    """
    APPLY CUSTOM FUNCTIONS TO GET THE DATA IN THE DESIRED FORMAT
    """
    dataset_required['Resolution Time'] = dataset_required.apply(lambda row:find_work_hour_diff(row['Call Date'],row['Completion Date']),axis=1)

    """
    OUTLIER REMOVAL ONLY FROM TRAINING DATA
    """
    dataset_required['Resolution Time after removing outlier'] = dataset_required.apply(lambda row: find_mean(dataset_required,row['Priority'],row['Category'],row['Subcategory'],row['Ticket Type']),axis=1)
    dataset_required['Final Resolution Time'] = dataset_required.apply(lambda row : find_final_rt(row['Resolution Time'],row['Resolution Time after removing outlier']),axis=1)
    dataset_required = fill_missing_month(dataset_required)
    dataset_required = fill_missing_sub(dataset_required,df)
    dataset_required = fill_missing_priority(dataset_required,df)

    """
    REMOVING THE COLUMNS WHICH ARE NOT REQUIRED FOR THE PREDICTION
    """
    dataset_required.drop(columns=['Call Date','Completion Date','Resolution Time','Resolution Time after removing outlier'],inplace=True)

    """
    ONE-HOT ENCODING OF CATEGORICAL VARIABLES
    """
    # Convert categorical variables into dummy variables
    # Priority, Category, Subcategory, and Ticket Type columns will be one-hot encoded
    progress_modified = pd.get_dummies(dataset_required, columns=['Priority', 'Category', 'Subcategory','month','Ticket Type'])


    """
    FEATURE(INPUT)-LABEL(OUTPUT) SEPARATION
    """
    # Set the output of label column
    label_column= 'Final Resolution Time'
    
    
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
    #     regressor.load_model(f"RTP_model.json")
    # else:
    #     # Creating an instance of XGBoost Regressor Model class
    regressor=XGBR()
    
    # Initialize GridSearchCV with the regressor (e.g., XGBoost) and the parameter grid
    # cv=5 specifies 5-fold cross-validation
    search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)
    
    # Fit the model with every combination of the above values
    # search.best_params_ returns the best combination of hyperparameters found during the search
    # print("The best hyperparameters are ", search.best_params_)

    # Creating an XGBoost regressor with the best parameters obtained from grid search
    regressor = XGBR(learning_rate=search.best_params_["learning_rate"],
                                n_estimators=search.best_params_["n_estimators"],
                                max_leaves=search.best_params_["max_leaves"],
                                max_depth=search.best_params_["max_depth"],
                                eval_metric='rmse')


    
        
    # Fitting the regressor to the training data
    regressor.fit(X_train, y_train)

    return regressor

def find_unique_ticket_type(dataset_latest_ticket):
    return dataset_latest_ticket['Ticket Type'].value_counts().index.tolist()


def rtp_main(dataset_latest_ticket,model_dir_path='.',model_file_name='Monthly_',test_size=0.3,random_state=23,\
    table_name='ResolutionTimePredictionOutput',\
    logfilename=f"rtp{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",test=False):

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
    
    dataset_latest_ticket['Call Date'] = pd.to_datetime(dataset_latest_ticket['Call Date'])
    dataset_latest_ticket['Completion Date'] = pd.to_datetime(dataset_latest_ticket['Completion Date'])
    
    unique_ticket_types = find_unique_ticket_type(dataset_latest_ticket)
    dataset_latest_ticket['Priority'] = dataset_latest_ticket['Priority'].apply(priority_conv)
    print(dataset_latest_ticket['Priority'].value_counts().index.tolist())
    for ticket_type in unique_ticket_types:
        print(ticket_type)
        X,y=data_preprocessing(dataset_latest_ticket,ticket_type)
        logger.info("Data was preprocessed.")

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
        regressor.save_model(f'{model_file_name}_rtp_{ticket_type}.json')
        shutil.move(f'{model_file_name}_rtp_{ticket_type}.json', rf'{model_dir_path}/{model_file_name}_rtp_{ticket_type}.json')  # move model JSON file

        logger.info(f"Model was saved in {model_dir_path} directory.")

if __name__=='__main__':
    rtp_main(df1, model_dir_path=r'.')

    
