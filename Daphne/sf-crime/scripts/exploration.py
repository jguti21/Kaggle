import pandas as pd
import os
import seaborn as sns
os.getcwd()
# READ THE CSV
df = pd.read_csv("./data/train.csv")

df.columns
df.head(2)

len(df) # 878 049 observations

# Let's do this thing where they say that you should not look at the test sample when you explore
from sklearn.model_selection import train_test_split
X = df.drop(columns="Category")
y = df["Category"].copy()
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
train, test = train_test_split(df, test_size=0.2)

###### Exploring the dependent variable
train["Category"].unique()
len(train["Category"].unique()) # 39 catégories

import matplotlib.pyplot as plt
sns.countplot(x="Category", data=train)
plt.xticks(rotation=50)

freq_cat = pd.DataFrame(train["Category"].value_counts(),
                        columns = ["Category", "freq"]).sort_values("freq")

train.columns
from pandas.api.types import DateDtype
# Month and categories
train["Dates"] = pd.to_datetime(train["Dates"])
train["Dates"].head(2)

train["Month"] = pd.DatetimeIndex(train["Dates"]).month
train.groupby(['Month', 'Category']).size().unstack().plot(kind='bar', stacked=True)


# Years
train["Year"] = pd.DatetimeIndex(train["Dates"]).year
train.groupby(['Year', 'Category']).size().unstack().plot(kind='bar', stacked=True)

# Evolution of each category in time
train["short_date"] = train["Dates"].dt.to_period('M')
train.groupby(["short_date", "Category"]).size().unstack().plot(kind='line')

categories = list(train.Category.unique())
sns.set_style("darkgrid")
for x in categories:
    df2 = pd.DataFrame(
        train[train["Category"] == x].groupby("short_date").size(),
        columns = ["count"])
    df2.reset_index(inplace=True)

    save_plots = sns.tsplot(x = "short_date",
               y = "count",
               data = df2).savefig("./plots/" + x + "intime.png")


sns.lineplot(x="short_date", y="count", data=df2)
import plotly.express as px
fig = px.line(df2, x='short_date', y='count')

# Hours
import datetime
train["hour"] = train["Dates"].split(" ", 1)[1]
train["hour"].head(2)
#### First model
from sklearn import tree
clf = tree.DecisionTreeClassifier()

X_train = train.drop(["Category", "short_date"], axis =1 )
X_train.columns
X_train.dtypes

y_train = train["Category"].copy()

clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)