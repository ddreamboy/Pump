import pandas as pd
import numpy as np
import time
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")

# https://practicaldatascience.co.uk/machine-learning/how-to-use-the-isolation-forest-model-for-outlier-detection

filename = 'sensor3.csv'
data = pd.read_csv(filename)

data_broken = data.loc[data['machine_status'].isin(['BROKEN'])] # оставляем только показания при состоянии BROKEN
data = data.fillna(data.median())
broken_ids = data_broken.index.values # получаем список индексов поломок
broken_count = len(broken_ids) # получаем количество поломок

data = data.loc[~data['machine_status'].isin(['BROKEN'])] # убираем значения при BROKEN
data_ = data.drop(columns = ['ID', 'timestamp', 'time_weekDays', 'time_dayTimes', 'machine_status'], axis = 1) # оставляем в датафрейме только значения сенсоров
# data_['timestamp'] = pd.to_datetime(data_['timestamp'])

# start = 1 # задаем начальное значение среза, так как используем .iloc указываем единицу
# end = broken_ids[broken_count - 2] # задаем конечное значение среза
# X_train = data_.iloc[start:end, :] # выполняем срез по заданному диапазону
X = data_.iloc[:, :]
X = X.transpose()
# start = broken_ids[broken_count - 2] # задаем начальное значение среза, так как используем .iloc указываем единицу
# end = broken_ids[broken_count - 1] # задаем конечное значение среза
# X_test = data_.iloc[start:end, :] # выполняем срез по заданному диапазону

data_broken = data_broken.drop(columns = ['ID', 'timestamp', 'time_weekDays', 'time_dayTimes', 'machine_status'], axis = 1)
# data_broken['timestamp'] = pd.to_datetime(data_broken['timestamp'])
Y = data_broken
Y = Y.transpose()
# Y_train = data_broken.drop(labels = [broken_ids[-1]], axis = 0)
# Y_test = data_broken.iloc[broken_count - 1, :]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressors = {
    "XGBRegressor": XGBRegressor(random_state=1)
}

df_models = pd.DataFrame(columns=['model', 'run_time', 'rmse', 'rmse_cv'])

for key in regressors:

    start_time = time.time()

    regressor = regressors[key]
    model = regressor.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    scores = cross_val_score(model, 
                             X_train, 
                             Y_train,
                             scoring="neg_mean_squared_error", 
                             cv=10)

    row = {'model': key,
           'run_time': format(round((time.time() - start_time)/60,2)),
           'rmse': round(np.sqrt(mean_squared_error(Y_test, y_pred))),
           'rmse_cv': round(np.mean(np.sqrt(-scores)))
    }

    df_models = df_models.append(row, ignore_index=True)

print(df_models.head())

iforest = IsolationForest(bootstrap=True,
                          contamination=0.01, 
                          max_features=10, 
                          max_samples=10, 
                          n_estimators=1000, 
                          n_jobs=-1,
                         random_state=1)
Y_predict = iforest.fit_predict(X_train)