import os
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
# from user_panel.models import UserPanel
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
import holidays as pyholidays 

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

    return bdiff.hours


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


def last_digit_as_number(s):
    # Extract the last character from the string and convert it to an integer
    return int(s[-1])

def find_actual_resolution_time(df,priority,category, subcategory):
    print(df[(df['Priority'] == priority) & (df['Category'] == category) & (df['Subcategory'] == subcategory)]['Resolution Time'])
    return df[(df['Priority'] == priority) & (df['Category'] == category) & (df['Subcategory'] == subcategory)]\
            ['Resolution Time'].mean()
    

def find_months(start_date,end_date):
    # Example: Create a DataFrame with a date range
    start_date = start_date
    end_date = end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame with the date range
    df = pd.DataFrame({'Date': date_range})

    # Extract Year and Month from the 'Date' column
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Create a new DataFrame with unique Year-Month combinations
    unique_year_month_df = df[['Year', 'Month']].drop_duplicates().sort_values(by=['Year', 'Month'])

    # Display the result
    return unique_year_month_df.to_dict(orient='records')
            
def rtp_infer(df,ticket_type,start_date='2023-01-01',end_date='2023-06-30',model_dir_path='.'):
    # df=df[df['Ticket Type']==ticket_type]
    df.replace('',np.nan,inplace=True)
    df.replace(' ',np.nan,inplace=True)

    df.replace('nan',np.nan,inplace=True)
    df.dropna(subset=['Call Date','Completion Date','Priority','Category','Subcategory'],inplace=True)

    dataset_ticket = df[['Call Date','Completion Date','Priority','Category','Subcategory']].copy()
    dataset_ticket['Priority'] = dataset_ticket['Priority'].apply(priority_conv)
    # dataset_ticket['Resolution Time'] = dataset_ticket.apply(lambda row : find_work_hour_diff(row['Call Date'],row['Completion Date']),axis=1)
    # print(dataset_ticket.head())
    unique_priority = dataset_ticket['Priority'].value_counts().index.tolist()
    unique_category = dataset_ticket['Category'].value_counts().index.tolist()
    unique_subcategory = dataset_ticket['Subcategory'].value_counts().index.tolist()
    unique_category_subcategory = dataset_ticket[['Category','Subcategory']].value_counts().index.tolist()
    unique_priority = sorted(unique_priority, key=last_digit_as_number)
    unique_category = sorted(unique_category)
    unique_subcategory = sorted(unique_subcategory)
    

    input_df = pd.DataFrame()
    df2 = dataset_ticket[['Priority','Category','Subcategory']].copy()
    output_df = pd.DataFrame()
    dict_encoded = {}
    dict_encoded['year'] = {0:0}
    columns = ['Priority','Category','Subcategory']
    for column in columns:
        if 'Pri' in column:
            final = unique_priority
        else:
            final = sorted(df2[column])
        for i in final:
            # print(column,'_',i,sep='')
            dict_encoded[column + "_" + i.strip().strip()] = {0: 0}

    for i in range(1,13):
        dict_encoded['month_'+str(i)] = {0:0}
       
    # for i in range(1,32):
    #     dict_encoded['day_'+str(i)] = {0:0}  

    dict_encoded['year'] = {0:0}

    input_df = pd.DataFrame(dict_encoded)
    output_df = pd.DataFrame(columns = ['Year','Call Month','Ticket Type','Priority','Category','Subcategory','Predicted Time']) 
    date_range = pd.date_range(start=start_date, end=end_date, freq='D') 
    
    predicted_time = 0
    model = XGBRegressor()
    model.load_model(model_dir_path)
    month_year_dict = find_months(start_date,end_date)
    for priority in unique_priority:
        for category,subcategory in unique_category_subcategory:
            if category is not None:
                input_df[f"Category_{category}"][0] = 1
            if priority is not None:
                input_df[f"Priority_{priority}"][0] = 1
            if subcategory is not None:
                input_df[f"Subcategory_{subcategory}"][0] = 1
            for val in month_year_dict:
                month = val['Month']
                year = val['Year']
                input_df['year'][0] = year
                input_df[f'month_{month}'][0] = 1
                input_df[f'Ticket Type_{ticket_type}']=1
                pridicted_time = model.predict(input_df[model.feature_names_in_])[0]
                if (predicted_time < 0):
                    predicted_time = 0
                new_row = {
                    'Year':year, 
                    'Call Month':month,
                    'Ticket Type':ticket_type,
                    'Priority':priority,
                    'Category':category,
                    'Subcategory':subcategory,
                    'Predicted Time':pridicted_time
                }
                output_df.loc[int(len(output_df))] = new_row
                input_df[f'month_{month}'][0] = 0
            input_df[f"Subcategory_{subcategory}"][0] = 0  
            input_df[f"Category_{category}"][0] = 0  
        input_df[f"Priority_{priority}"][0] = 0
    print('Its a SUCCESS')
    print(output_df)    
    output_df.loc[output_df['Predicted Time'] < 0, 'Predicted Time'] = 7
    return output_df
    