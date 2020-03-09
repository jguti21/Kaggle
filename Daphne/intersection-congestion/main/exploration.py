import os
import pandas as pd
import seaborn as sns
import folium

df_train = pd.read_csv('./data/train.csv')

df_train.columns

# Create a map of the intersections
unique_intersection = df_train.drop_duplicates(["Longitude", "Latitude"])

m = folium.Map(
    tiles="Stamen Toner",
    zoom_start=2
)

for i in range(0, len(unique_intersection)):
    folium.Marker([unique_intersection.iloc[i]["Latitude"],
                  unique_intersection.iloc[i]["Longitude"]],
                  popup=unique_intersection.iloc[i]["Path"]).add_to(m)

m.save("./maps/intersection_location.html")

## IF I understand correctly we have month/hour aggregated values
# for multiple intersections
# The variables to be predicted are the density distribtuion
# for the distance before first stop and the total time stropped

len(df_train[df_train["IntersectionId"] == 0])
len(df_train[df_train["IntersectionId"] == 1])

len(df_train.drop_duplicates(["IntersectionId"])) # In the end 2559 unique intersections

# Hmmm the key is probably to summarize the information

# I want to plot the average per city
# avg_city_time = df_train.groupby("City")["TotalTimeStopped_p20", "TotalTimeStopped_p40",
#                                 "TotalTimeStopped_p50", "TotalTimeStopped_p60",
#                                 "TotalTimeStopped_p80"].mean()
# avg_city_time = avg_city_time.unstack()
# avg_city_time = avg_city_time.reset_index()
#
# avg_city_time.head(5)
#
# sns.lineplot(x="level_0", y=0,
#              hue="City",
#              data=avg_city_time).get_figure().savefig("./charts/avg_timestropped_city")
#
#
# # I want to plot the average per hour
# avg_hour_time = df_train.groupby("Hour")["TotalTimeStopped_p20", "TotalTimeStopped_p40",
#                                 "TotalTimeStopped_p50", "TotalTimeStopped_p60",
#                                 "TotalTimeStopped_p80"].mean()
# avg_hour_time = avg_hour_time.unstack()
# avg_hour_time = avg_hour_time.reset_index()
#
# avg_hour_time.head(5)
#
# sns.lineplot(x="level_0", y=0,
#              hue="Hour",
#              data=avg_city_time).get_figure().savefig("./charts/avg_timestropped_city")

# Estimating the function from the quantiles: https://stats.stackexchange.com/questions/6022/estimating-a-distribution-based-on-three-percentiles

# So we have to model the time stopped
# multiple entries per intersection:
# - per entry/exit
# - per month
# - per hour

# I want to model the average time I will be stopped (I mean the one of all the ppl in that month) at that intersection
# I want the distribution more than the average

# What could be relevant?
# - this tipic intersection (closeness of the city centre, POI around, intersection between big/small roads)
# - the time (hour yes, month ?)
# - the city (Atlanta is famous for its casinos)
# - my destination (?)

# Let's try something: plot the densities of the target variables
# First by looking at their distribution in time
test = df_train[df_train["IntersectionId"] == 2] # I changed for 0, 1, 2 to plot the graph on the top of each other
sns.kdeplot(test['TotalTimeStopped_p80'], shade=True)
# We can see that the 80 percentile is not normal at all and it changes a lot from one intersection to another

# Second looking at one point in time
test = df_train[df_train["IntersectionId"] == 0]
test = test[(test["Month"] == 8) & (test["Hour"] == 12)] # IntersectionId is not unique per City!!!!
test = test[test["City"] == "Philadelphia"] # Multiple entries for the same month/hour but one is week-end and the
# other one is not. Multiple years?!
# There is a time dimension that is missing!!!!! Probably some entries are driven by only one truck, hence
# we don't always have the same entries and exits offer for different timings.

# Looking with a filter on the path
test = df_train[(df_train["IntersectionId"] == 2) & (df_train["Path"] == 'Glenwood Avenue Southeast_W_Glenwood Avenue Southeast_W')] # I changed for 0, 1, 2 to plot the graph on the top of each other
sns.kdeplot(test['TotalTimeStopped_p80'], shade=True)
### AHHHH that looks way more normal !

test = df_train[(df_train["IntersectionId"] == 0)
                & (df_train["Path"] == 'Marietta Boulevard Northwest_SE_Marietta Boulevard Northwest_SE')]
sns.kdeplot(test['TotalTimeStopped_p80'], shade=True)
### AHHH that does not at all ....

### I UNDERSTOOD, the records are taken during the week and the weekend. Each is duplicated per month and hour.


## Why not a time continuous markov chain for the estimation?
# https://math.stackexchange.com/questions/876789/continuous-time-markov-chains-is-this-step-by-step-example-correct/880405
# Continuous time markov chain (or its hidden version) <= good challenge to implement it from scratch


var_time_stopped =["TotalTimeStopped_p20", "TotalTimeStopped_p40", "TotalTimeStopped_p50", "TotalTimeStopped_p60",
                   "TotalTimeStopped_p80"]
test = test[var_time_stopped]
test.head()
