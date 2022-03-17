import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime, timedelta

filename  = 'sensor2.csv'
data = pd.read_csv(filename)

filename = 'sensor_base_test.csv'
database = pd.read_csv(filename)

# db_last_id = len(database.index) - 1
# delta_time = timedelta.total_seconds(pd.to_datetime(data.loc[data_id, 'timestamp']) - pd.to_datetime(database.loc[db_last_id, 'timestamp']))
# print(delta_time)
    
def value_processing(database, data):
    pass

def last_days():
    filename = 'sensor_base_test.csv'
    database = pd.read_csv(filename)
    if len(database.index) >= 10082:
        diapason = [i for i in range(0, len(database.index) - 1440)] # 7 дней = 10080
        database_viz = database.drop(labels = diapason)
        database_viz.to_csv('database_viz.csv', encoding='utf-8', index=False)
        
while True:
    data_id = len(database.index)
    database = database.append(data.loc[data_id, :], ignore_index = True)
    for i in tqdm(range(0, 52)): # проходимся по всем сенсорам
        if i < 10: 
            string = '0' + str(i)
        else:
            string = str(i)
        sensor = 'sensor_' + string

        if i not in [15, 50]:
            if np.isnan(data.loc[data_id, sensor]) or data.loc[data_id, sensor] == 0:
                day = data.loc[data_id, 'time_weekDays']
                data_com = database.loc[data['time_weekDays'].isin([day])] # оставляем показания только за интересующий нас день
                data_com = data_com.loc[ : , sensor] # берем показания нужного нам сенсора
                median = data_com.median() # берем медиану
                database.loc[data_id, sensor] = median
                print('\n')
                print(f'ID: {data_id}')
                print(f'Название сенсора: {sensor}')
                timestamp = database.loc[data_id, 'timestamp']
                print(f'Дата и время: {timestamp}')
                print(f'Значение: {database.loc[data_id, sensor]}')
                print('Статус значения: Не пришло, заменено')
                print('\n')
    database.to_csv('sensor_base_test.csv', encoding='utf-8', index=False)
    last_days()
    time.sleep(1)