# First step, bringing in data
import pandas as pd
data = pd.read_csv('sphist.csv')

# Convert the date column into a datetime object
data['Date'] = pd.to_datetime(data['Date'])

# Order the data set by date, ascending
data = data.sort_values(by = ['Date'], ascending=True)
print(data[0:5])

# Getting some indicators, storing as new cols
data['day_30'] = 0
data['day_30_sd'] = 0
data['day_30_msd'] = 0
data['day_5_vol'] = 0
data['day_30_vol'] = 0
data['vol_ratio'] = 0
# 30 day average
data['day_30'] = data['Close'].rolling(30).mean()
data['day_30'] = data['day_30'].shift()
# 30 say standard deviation
data['day_30_sd'] = data['Close'].rolling(30).std()
data['day_30_sd'] = data['day_30_sd'].shift()
# ratio between the two
data['day_30_msd'] = data['day_30'] / data['day_30_sd']
# 5 and 30 day volumes
data['day_5_vol'] = data['Volume'].rolling(5).mean()
data['day_5_vol'] = data['day_5_vol'].shift()
data['day_30_vol'] = data['Volume'].rolling(30).mean()
data['day_30_vol'] = data['day_30_vol'].shift()
# Ratio between the two
data['vol_ratio'] = data['day_5_vol'] / data['day_30_vol']


# Remove any rows that fall before Feb 3, 1950
# These are NAs for our 30 day moving averages
from datetime import datetime
data = data[data['Date'] > datetime(year = 1950, month = 2, day = 3)]
# Get rid of the NA rows
data = data.dropna(axis=0)

# Make two new datasets, train and test.
# train is before Jan 01, 13 and test is after
train = data[data['Date'] < datetime(year = 2013, month = 1, day = 1)]
test = data[data['Date'] >= datetime(year = 2013, month = 1, day = 1)]

# Modeling - going to use MAE as an error metric
from sklearn.linear_model import LinearRegression
# Instantiate a Linear Regression
lr = LinearRegression()

# Select our columns (some redundancy, I know), and train the model
cols = ['day_30', 'day_30_msd', 'day_30_sd']
lr.fit(train[cols], train['Close'])

# Get our predictions
predictions = lr.predict(test[cols])

# Get the mean absolute error of our model
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test['Close'], predictions)
print(mae)

# Seems we're off by about $31 generally.

# Let's add two new indicators, see if we can get a bit better. I've gone back up in the code and added three more calculations, 5 day average volume, 30 day average volume and the ratio between the two, before we deleted NAs and split the data. I'm going to use 30 day avg and the ratio in this next model.

# New model classs, columns, and fit
lr2 = LinearRegression()
cols = ['day_30', 'day_30_msd', 'day_30_sd', 'day_30_vol', 'vol_ratio']
lr2.fit(train[cols], train['Close'])

# Second set of predictions
predictions2 = lr2.predict(test[cols])

# New MAE
mae2 = mean_absolute_error(test['Close'], predictions2)
print(mae2)

# Lol, with these two new predictors we got a bit more accurate. About 30 cents more accurate.











