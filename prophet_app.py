# Dependencies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')

# Read in and print data
aep = pd.read_csv('AEP_hourly.csv',index_col=[0], parse_dates=[0])
print("AEP Data ----------------------------")
print(aep.head())

# Plotting the data over the 14 years. THIS SHOWS UP BUT SEEMS TO LOCK UP THE APPs

#color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
#_ = aep.plot(style='.', figsize=(15,5), color=color_pal[0], title='AEP Energy Use Over Time')
#plt.show()

# Function to create features based on time

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(aep, label='AEP_MW')

features_and_target = pd.concat([X, y], axis=1)

print("Features and Target ----------------------------")
print(features_and_target.head())

# Seaborn fancy plots with lots of colors. Will this crash? It might crash.

sns.pairplot(features_and_target.dropna(),
            hue= 'hour',
            x_vars=['hour','dayofweek','year','weekofyear'],
             y_vars='AEP_MW',
             height=5,
             plot_kws={'alpha':0.1, 'linewidth':0}
            )
plt.suptitle('Power usage')

plt.show()

# Train/Test Split

split_date = '01-Jan-2015'
aep_train = aep.loc[aep.index <= split_date].copy()
aep_test = aep.loc[aep.index > split_date].copy()

_ = aep_test \
    .rename(columns={'AEP_MW': 'TEST SET'}) \
    .join(aep_train.rename(columns={'AEP_MW': 'TRAINING SET'}), how='outer') \
    .plot(figsize=(15,5), title='AEP East', style='.')
    
print("Train/Test Split ----------------------------")
print(aep_train.reset_index().rename(columns={'Datetime' : 'ds', 'AEP_MW': 'y'}).head())

# Initializing the Prohpet Model and fitting the training data

model = Prophet()
model.fit(aep_train.reset_index().rename(columns={'Datetime':'ds', 'AEP_MW':'y'}))

aep_test_fcst = model.predict(df=aep_test.reset_index().rename(columns={'Datetime':'ds'}))

f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(aep_test_fcst, ax=ax, title="AEP Prophet Model Predictions')

fig = model.plot_components(aep_test_fcst)

plt.show()

# Plot the forecast with the actuals

f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(aep_test.index, aep_test['AEP_MW'], color='r')
fig = model.plot(aep_test_fcst, ax=ax)

# Create a time period of predictions

# Variable to store user inputs
begin_month = input("Enter start of time period in 01-01-2000 format: ")
end_month = input("Enter end of time period in 01-01-2000 format: ")

# Plot the forecast with the actuals
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(aep_test.index, aep_test['AEP_MW'], color='r')
fig = model.plot(aep_test_fcst, ax=ax)
ax.set_xbound(lower=begin_month, upper=end_month)
ax.set_ylim(0, 60000)
plot = plt.suptitle('January 2015 Forecast vs Actuals')
plt.show()

# Single week of predictions

# Plot the forecast with the actuals

#f, ax = plt.subplots(1)
#f.set_figheight(5)
#f.set_figwidth(15)
#ax.scatter(aep_test.index, aep_test['AEP_MW'], color='r')
#fig = model.plot(aep_test_fcst, ax=ax)
#ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
#ax.set_ylim(0, 60000)
#plot = plt.suptitle('First Week of January Forecast vs Actuals')
#plt.show()

# Error metrics

#mean_squared_error(y_true=aep_test['AEP_MW'],
#                   y_pred=aep_test_fcst['yhat'])
#                   
#print('mean squared error ------------------------------')
#print(mean_squared_error)
    

