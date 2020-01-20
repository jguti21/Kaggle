import re
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from math import ceil
import os


trainortest = "test"

def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom / 7.0))

def reducing_address(address):
    list_address = address.split(" ")
    final_address = []
    for word in list_address:
        if (len(word) == len([letter for letter in word if letter.isupper()]))\
                and word != "ST" and word != "AV":
            final_address.append(word)
    return "".join(final_address)


# cat_var = ['Category', 'PdDistrict', 'street']
cat_var = ['PdDistrict', 'street']

# Read the data
df = pd.read_csv("./data/" + trainortest + ".csv", parse_dates=['Dates'])
# len(df) # 878 049 observations

# No null values

# Duplicates
# df.duplicated().sum() # 2323 exactly
df.drop_duplicates(inplace=True)

# Outliers removal
df = df[(df.X < -121.000) & (df.Y < 40.000)]  # this is some location in the pacific

# Address
# Shortening the values
df["street"] = df.Address.apply(reducing_address)

# Encoding categories
encoder = LabelEncoder()
for var in cat_var:
    try:
        df[var] = encoder.fit_transform(df[var].astype(str))

    except:
        pass

# Dates
df['Date'] = pd.to_datetime(df['Dates'].dt.strftime('%Y-%m-%d'),
                            format='%Y-%m-%d')
df['Hour'] = df.Dates.dt.hour
df['Day'] = df.Dates.dt.day
df['Month'] = df.Dates.dt.month
df['Year'] = df.Dates.dt.year
df["DayOfWeek"] = df.Dates.dt.dayofweek  # Monday=0, Sunday=6.
df["Fri"] = np.where(df.DayOfWeek == "Friday", 1, 0)
df["Sat"] = np.where(df.DayOfWeek == "Saturday", 1, 0)
df["WeekOfMonth"] = df.Dates.map(week_of_month)
df["Weekend"] = df["Fri"] + df["Sat"]

# Public holidays
df["PH"] = np.where(
    ((df.Day == 1) & (df.Month == 1))  # New Year
    | ((df.DayOfWeek == 0) & (df.Month == 1) & (df.WeekOfMonth == 3))  # Luther's day (3rd Monday of January)
    | ((df.DayOfWeek == 0) & (df.Month == 2) & (df.WeekOfMonth == 3))  # President's day (3rd Monday in February)
    | ((df.DayOfWeek == 0) & (df.Month == 5) & (df.WeekOfMonth > 3))  # Memorial's day (last Monday of May)
    | ((df.Day == 4) & (df.Month == 7))  # Independence day
    | ((df.DayOfWeek == 0) & (df.Month == 9) & (df.WeekOfMonth == 1))  # Labor Day (1st Monday in September)
    | ((df.DayOfWeek == 0) & (df.Month == 10) & (df.WeekOfMonth == 2))  # Columbus Day (2nd Monday in October)
    | ((df.Day == 11) & (df.Month == 11))  # Veteran's day
    | ((df.DayOfWeek == 3) & (df.Month == 11) & (df.WeekOfMonth > 3))  # Thanksgiving (Last Thursday in November)
    | ((df.Day == 25) & (df.Month == 12)),  # Christmas
    1,
    0
)

# Weather
weather_data = pd.read_csv("./data/weather san_francisco_Jan2003-Dec15.csv")
weather_data["Date"] = weather_data["Date"].str.replace(" ", "")
weather_data["Date"] = pd.to_datetime(weather_data["Date"], format='%Y-%m-%d')

weather_data.columns = ['t_max', 't_avg', 't_min', 'dew_max', 'dew_avg', 'dew_min', 'hum_max',
                        'hum_avg', 'hum_min', 'wind_max', 'wind_avg', 'wind_min', 'pres_max',
                        'pres_avg', 'pres_min', 'percip', 'Date']
weather_data = weather_data.drop(
    [
        't_max', 't_min', 'dew_max', 'dew_avg', 'dew_min', 'hum_max',
        'hum_min', 'wind_max', 'wind_min', 'pres_max', 'pres_avg',
        'pres_min', 'percip'
    ], axis=1)

df = pd.merge(df, weather_data, on="Date", how="left")


# Keep only relevant columns
if trainortest == "train":
    df = df.drop(["Dates", "Fri", "Sat", "WeekOfMonth", "Descript",
                  "Address", "Resolution", "Date"], axis=1)
elif trainortest == "test":
    df = df.drop(["Dates", "Fri", "Sat", "WeekOfMonth",
                  "Address", "Date"], axis=1)

# Save
df.to_csv("./data/" + trainortest + "_prepared.csv", index=False)
