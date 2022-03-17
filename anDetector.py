import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# https://practicaldatascience.co.uk/machine-learning/how-to-use-the-isolation-forest-model-for-outlier-detection

filename = 'sensor3.csv'
data = pd.read_csv(filename, index_col = 'timestamp')

data = data.loc[~data['machine_status'].isin(['BROKEN'])] # убираем значения при BROKEN
data = data.drop(columns = ['ID', 'time_weekDays', 'time_dayTimes', 'machine_status'], axis = 1) # оставляем в датафрейме только значения сенсоров
data.index = pd.to_datetime(data.index)
data = data.fillna(data.median())

x = data.values

model = IsolationForest(contamination=0.1)
model.fit_predict(x)
decFunction = model.decision_function(x)
data['outliers'] = decFunction

plt.plot(data['outliers'])
plt.axhspan(-0.2, 0, alpha=0.2, color='red')
plt.show()