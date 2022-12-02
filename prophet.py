import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
from datetime import datetime
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import time
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


data_complete = pd.read_csv('data/walmart_cleaned.csv')


all_store_dept = data_complete[['Store','Dept']].drop_duplicates()

all_store_dept["combined"] = all_store_dept["Store"].astype(str) + "_" + all_store_dept["Dept"].astype(str)
list_required_store_dept = all_store_dept["combined"].to_list()

data_complete['store_dept_combined'] = data_complete["Store"].astype(str) + "_" + data_complete["Dept"].astype(str)

data_complete = data_complete[data_complete['store_dept_combined'].isin(list_required_store_dept)]\
                .reset_index(drop=True)


text_file = open("error_prophet.txt", "w")


a_time = time.time()
count = 0
for i in list_required_store_dept:
    try:
        count+=1
        if(count%100==0): print(count)
        data_copy = data_complete[data_complete['store_dept_combined']==i]\
                    .reset_index(drop=True).copy()
        data_copy = data_copy[['Weekly_Sales','Date','IsHoliday','Temperature', 'Fuel_Price','MarkDown1', 'MarkDown2', 
                               'MarkDown3','MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']]
        data_copy.columns = ['y', 'ds','IsHoliday','Temperature', 'Fuel_Price','MarkDown1', 'MarkDown2', 'MarkDown3',
               'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']
        def fit_predict_model(dataframe, interval_width = 0.99, changepoint_range = 0.9, changepoint_prior_scale=0.5):
            m = Prophet(daily_seasonality = False, yearly_seasonality = False, weekly_seasonality = False,
                        seasonality_mode = 'multiplicative', 
                        interval_width = interval_width,
                        changepoint_range = changepoint_range, changepoint_prior_scale = changepoint_prior_scale)

            m.add_regressor('IsHoliday')
            m.add_regressor('Temperature')
            m.add_regressor('Fuel_Price')
            m.add_regressor('MarkDown1')
            m.add_regressor('MarkDown2')
            m.add_regressor('MarkDown3')
            m.add_regressor('MarkDown4')
            m.add_regressor('MarkDown5')
            m.add_regressor('CPI')
            m.add_regressor('Unemployment')

            m = m.fit(dataframe)

            forecast = m.predict(dataframe)
            forecast['fact'] = dataframe['y'].reset_index(drop = True)

            return forecast

        pred = fit_predict_model(data_copy)

        def detect_anomalies(forecast):
            forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
            #forecast['fact'] = df['y']

            forecasted['anomaly'] = 0
            forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
            forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

            #anomaly importances
            forecasted['importance'] = 0
            forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
                (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
            forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
                (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']

            return forecasted

        pred = detect_anomalies(pred)
        pred = pred[pred['anomaly'].isin([1,-1])].reset_index(drop = True)
    
        pred.drop(columns=['trend','yhat_lower','yhat_upper'], axis=1, inplace=True)

        pred.to_csv('prophet/'+ i +'.csv', index=False)
        
    except:
        count+=1
        text_file.write(i+"\n")
            
text_file.close()
print("Total time in minutes:\t", (time.time()- a_time)/60)