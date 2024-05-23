import datetime
import os
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json
import warnings
from multiprocessing import Pool



def load_model(model_path):
    with open(model_path, "r") as fin:
        model_json = fin.read()
    model = model_from_json(model_json)
    return model

def date_filler(dataset,date_col):
    # Create a range of dates
    dataset[date_col]=pd.to_datetime(dataset[date_col])
    dataset.dropna(inplace=True)
    
    date_range = pd.date_range(start=dataset[date_col].min(), end=dataset[date_col].max())
    # Create a new DataFrame with all dates and fill missing values with 0
    new_dataset = dataset.set_index(date_col).reindex(date_range).fillna(0)

    # Reset the index to obtain the final DataFrame
    new_dataset = new_dataset.reset_index()
    new_dataset.columns=[date_col]+list(new_dataset.columns[1:])
    return new_dataset

def infer(category_var,priority_var,ticket_type_var,subcategory_var,first_date,end_date,model_dir_path='.'):
    """
    Loads the approapriate Prophet model for given conditions and gives the volume forecast for the given daterange or date.

    The avilable valid combinations as per the latest given TopDesk data "All calls incl memos 27 Oct 2023.xlsx" is given below.

    Category                                     Priority
    Accounts Payable (PtP)        P1-P6, "all"
    Accounts Receivable (OtC) P1-4, "all"
    General Ledger (RtR)            P1-4, "all"
    Integrations                            P1-4, "all"
    Procurement                             P1-4, "all"
    System                                        P1-4, "all"
    Travel & Expenses                 P2-4, "all"
    User Management                     P2-P6, "all"
    all                                             P1-P6, "all"
    Note that there is no Pririty 5 or 'P5'.

    Args:
        category_var: The category you want to forecast the volume for. i
        priority_var: The priority you want to forecast the volume for. Valid inputs are 1,2,3,4,'P6' and check the above table for availability
                                    for the category you want.
        first_date:     The last date of the range you want to predict for preferably in DD-MM-YYYY format
        end_date:         The last date of the range you want to predict for preferably in DD-MM-YYYY format. It should be the same
                                    as first_date in case you want to forecast just for a single date.
        model_dir_path: In case the model json files are not in the current path, then provide the actual path sans the model json name here.

    Returns:
        A forecast dataframe with all its variables.
    """
    # lambda x:x.replace('SLASH','/')
    try:
        model_file_name = 'category_'+category_var+'_priority_'+priority_var+'_ticket_type_'+ticket_type_var+'_subcategory_'+subcategory_var+'_model.json'
        model_file_name=model_file_name.replace('/','SLASH')
        print(model_file_name)
        m=load_model(os.path.join(model_dir_path,model_file_name))
    except:
        model_file_name = 'category_'+category_var+'_priority_'+priority_var+'_tickettype_'+ticket_type_var+'_subcategory_'+subcategory_var+'_model.json'
        model_file_name=model_file_name.replace('/','SLASH')
        print(model_file_name)
        m=load_model(os.path.join(model_dir_path,model_file_name))
        
    future = m.make_future_dataframe(periods=246)
    if first_date==end_date:
        dates_dataset=pd.DataFrame({'ds':first_date},index=[0])
    else:
        dates_dataset=pd.DataFrame({'ds': pd.date_range(start=first_date, end=end_date, freq='D')})
    dates_dataset['cap']=m.history.cap[0]#(dataset.y.max()*3)+1
    dates_dataset['floor']=0
    forecast_dataset = m.predict(dates_dataset)
    forecast_dataset['cap']=m.history.cap[0]
    forecast_dataset['floor']=0
    forecast_dataset.loc[forecast_dataset['yhat']<forecast_dataset['floor'],'yhat']=forecast_dataset['floor']
    forecast_dataset.loc[forecast_dataset['yhat']>forecast_dataset['cap'],'yhat']=forecast_dataset['cap']
    #forecast_dataset['yhat']=forecast_dataset['yhat'].apply(np.around).astype('int')
    forecast_dict={}
    for j,i in enumerate(forecast_dataset.ds):
        forecast_dict[str(i.date())]=forecast_dataset.yhat[j]
    forecast_dataset.dropna(inplace=True)
    forecast_dataset['Day']=forecast_dataset['ds'].dt.day_name()
    return forecast_dataset[['ds','yhat','Day']] #forecast_dict



def forecast_curr(category_var, priority_var, ticket_type_var, subcategory_var, first_date='2023-03-01', end_date='2025-02-28', model_dir_path='media/tvp_models/'):
  forecast_df = infer(category_var,priority_var,ticket_type_var,subcategory_var,first_date,end_date,model_dir_path=model_dir_path)
  forecast_df.columns = ['Date','Predicted_Count','Day']
  forecast_df['Category'] = category_var
  forecast_df['Ticket Type'] = ticket_type_var
  forecast_df['Subcategory'] = subcategory_var
  forecast_df['Priority'] = priority_var
  return forecast_df

def worker(model_file, model_dir_path, first_date, end_date):
  name_split_list = model_file.split('_')
  category_var = name_split_list[1]
  priority_var = name_split_list[3]
  ticket_type_var = name_split_list[6]
  subcategory_var = name_split_list[8]
  try:
    forecast_curr_df = forecast_curr(category_var, priority_var, ticket_type_var, subcategory_var, first_date, end_date, model_dir_path)
    return forecast_curr_df[['Date', 'Category','Subcategory', 'Priority', 'Ticket Type', 'Day', 'Predicted_Count']]
  except:
    print(category_var, priority_var, ticket_type_var, subcategory_var)
    print(model_file)
    return None

def forecast_all3_parallel(first_date='2023-03-01', end_date='2025-02-28', model_dir_path='media/tvp_models/'):
  dataset = pd.DataFrame(columns=['Date', 'Category','Subcategory', 'Priority', 'Ticket Type', 'Day', 'Predicted_Count'])
  count = 0
  with Pool() as pool:
      results = pool.starmap(worker, [(mf, model_dir_path, first_date, end_date) for mf in os.listdir(model_dir_path)])
      for result in results:
          if result is not None:
              dataset = pd.concat([dataset, result])
  dataset['Subcategory'] = dataset.Subcategory.apply(lambda x:x.replace('SLASH','/'))
  return dataset

def forecast_all3(first_date='2023-03-01',end_date='2025-02-28',model_dir_path='media/tvp_models/'):
    dataset=pd.DataFrame(columns=['Date', 'Category','Subcategory', 'Priority', 'Ticket Type', 'Day', 'Predicted_Count']) #, 'Actual_Count'
    count=0
    for model_file in os.listdir(model_dir_path):
        name_split_list=model_file.split('_')
        # print(name_split_list)
        new_dict={}
        # for i in name_split_list:
            
        category_var=name_split_list[1]
        priority_var=name_split_list[3]
        ticket_type_var=name_split_list[6]
        subcategory_var=name_split_list[8]
        
        try:
            forecast_curr=infer(category_var,priority_var,ticket_type_var,subcategory_var,first_date=first_date,end_date=end_date,model_dir_path=model_dir_path)
            forecast_curr.columns=['Date','Predicted_Count','Day']
            forecast_curr['Category']=category_var
            forecast_curr['Ticket Type']=ticket_type_var
            forecast_curr['Subcategory']=subcategory_var
            forecast_curr['Priority']=priority_var
            dataset=pd.concat([dataset,forecast_curr[['Date', 'Category','Subcategory', 'Priority', 'Ticket Type', 'Day', 'Predicted_Count']]])
        except:
            print(category_var,priority_var,ticket_type_var,subcategory_var)
            print(model_file)
        count+=1
    #print(count)
    dataset['Subcategory']=dataset.Subcategory.apply(lambda x:x.replace('SLASH','/'))
    return dataset
    
import sys
if __name__=='__main__':
    # Turn off all warnings
    warnings.filterwarnings("ignore")
    df=forecast_all3_parallel(model_dir_path=sys.argv[1])
    print(df.shape)
    print(df.head())
    print(df.tail())
    df.to_csv('new_tvf_cstp.csv',index=False)
