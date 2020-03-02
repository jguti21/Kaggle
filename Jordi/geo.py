# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:10:30 2020

@author: gutierj
"""
#%% Packages and directory

#geo.py

import os
import pandas as pd    
import numpy as np
import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

#home
#os.chdir('C:/Users/jordi/Desktop/Work/Kaggle/Geo')

#work
os.chdir('C:/Users/gutierj/Desktop/Programming/Kaggle/Geo')


#Import data

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


#%% Summary and charts


#Summary function of the data

#https://www.kaggle.com/kabure/insightful-eda-modeling-lgbm-hyperopt

#understanding resumetable function
"""
summary = pd.DataFrame(df_train.dtypes,columns=['dtypes'])
summary = summary.reset_index() #reset index to add it as a column
summary['Name'] = summary['index']
summary = summary[['Name','dtypes']]    #remove index
summary['Missing'] = df_train.isnull().sum().values    
summary['Uniques'] = df_train.nunique().values
summary['First Value'] = df_train.loc[0].values  # all values in the first row
summary['Second Value'] = df_train.loc[1].values
summary['Third Value'] = df_train.loc[2].values

#add entropy  column

 a measure of "disorder" (the higher the entropy, the higher the disorder)
 
for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df_train[name].value_counts(normalize=True), base=2),2) 
        
"""


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary
        
resumetable(df_train)


# City count distribution plot

total = len(df_train)
plt.figure(figsize=(15,19))

plt.subplot(311)

# Countplot per city
g = sns.countplot(x="City", data=df_train)
g.set_title("City Count Distribution", fontsize=20)
g.set_ylabel("Count",fontsize= 17)
g.set_xlabel("City Names", fontsize=17)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.15)

plt.show()

#df_train mostly in Philadelphia


# Date features

#Count unique cities and hours (groupby)
tmp_hour = df_train.groupby(['City', 'Hour'])['RowId'].nunique().reset_index()



plt.subplot(211)

#countplot of hour per city
g = sns.countplot(x="Hour", data=df_train, hue='City', dodge=True)
g.set_title("Hour Count Distribution by Week and Weekend Days", fontsize=20)
g.set_ylabel("Count",fontsize= 17)
g.set_xlabel("Hours of Day", fontsize=17)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)

g.set_ylim(0, max(sizes) * 1.15)

plt.subplot(212)

#count per month
g1 = sns.countplot(x="Month", data=df_train, hue='City', dodge=True)
g1.set_title("Hour Count Distribution by Week and Weekend Days", fontsize=20)
g1.set_ylabel("Count",fontsize= 17)
g1.set_xlabel("Hours of Day", fontsize=17)
sizes=[]
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)

g1.set_ylim(0, max(sizes) * 1.15)

plt.subplots_adjust(hspace = 0.3)

plt.show()



# Entry heading and Exit heading

plt.figure(figsize=(15,12))

#Count % of unique entry heading

tmp = round(((df_train.groupby(['EntryHeading'])['RowId'].nunique() / total) * 100)).reset_index()

plt.subplot(211)

#use it to sort the chart of countplot of entry headings
g = sns.countplot(x="EntryHeading",
                  data=df_train,
                  order=list(tmp['EntryHeading'].values),
                  hue='ExitHeading', dodge=True)
g.set_title("Entry Heading by Exit Heading", fontsize=20)
g.set_ylabel("Count",fontsize= 17)
g.set_xlabel("Entry Heading Region", fontsize=17)
gt = g.twinx()

#points in the plot with tmp values
gt = sns.pointplot(x='EntryHeading', y='RowId', 
                   data=tmp, order=list(tmp['EntryHeading'].values),
                   color='black', legend=False)
gt.set_ylim(0, tmp['RowId'].max()*1.1)
gt.set_ylabel("% of Total(Black Line)", fontsize=16)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)

g.set_ylim(0, max(sizes) * 1.15)

plt.subplot(212)

#countplot of entryheading by city
g1 = sns.countplot(x="EntryHeading", order=list(tmp['EntryHeading'].values), 
                   data=df_train, hue='City')
g1.set_title("Entry Heading Distribution By Cities", fontsize=20)
g1.set_ylabel("Count",fontsize= 17)
g1.set_xlabel("Entry Heading Region", fontsize=17)
sizes=[]
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)

g1.set_ylim(0, max(sizes) * 1.15)

plt.subplots_adjust(hspace = 0.3)

plt.show()


#IntersectionID

plt.figure(figsize=(15,6))

#count top 45 intersectionIds with most counts
df_train.IntersectionId.value_counts()[:45].plot(kind='bar')
plt.xlabel("Intersection Number", fontsize=18)
plt.ylabel("Count", fontsize=18)
plt.title("TOP 45 most commmon IntersectionID's ", fontsize=22)

plt.show()

df_train.groupby(['IntersectionId', 'EntryHeading', 'ExitHeading'])['RowId'].count().reset_index().head()

# Exploring numerical features

t_stopped = ['TotalTimeStopped_p20',
             'TotalTimeStopped_p50', 
             'TotalTimeStopped_p80']
t_first_stopped = ['TimeFromFirstStop_p20',
                   'TimeFromFirstStop_p50',
                   'TimeFromFirstStop_p80']
d_first_stopped = ['DistanceToFirstStop_p20',
                   'DistanceToFirstStop_p50',
                   'DistanceToFirstStop_p80']


# Heatmap target features

#timestopped + distance to first stop
plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set', fontsize=22)
sns.heatmap(df_train[t_stopped + 
                     #t_first_stopped + 
                     d_first_stopped].astype(float).corr(),
            vmax=1.0,  annot=True)
plt.show()


#Distance to First Stop p20 and Total Time Stopped p20 have a high correlation.


# Scaling the target

#getting the min_max transformation to get clusterization and PCA features

from sklearn.preprocessing import minmax_scale

target_cols = t_stopped + d_first_stopped

for col in target_cols:
    df_train[col+str("_minmax")] = (minmax_scale(df_train[col], feature_range=(0,1)))
    
min_max_cols = ['TotalTimeStopped_p20_minmax', 'TotalTimeStopped_p50_minmax',
                'TotalTimeStopped_p80_minmax', 'DistanceToFirstStop_p20_minmax',
                'DistanceToFirstStop_p50_minmax', 'DistanceToFirstStop_p80_minmax']


#PCA
pca = PCA(n_components=3, random_state=5)

principalComponents = pca.fit_transform(df_train[min_max_cols])

principalDf = pd.DataFrame(principalComponents)

prefix='Target_PCA'
principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

df_train = pd.concat([df_train, principalDf], axis=1)
    
pca.explained_variance_ratio_[:2].sum()

#Scatterplot of cities by PCA

#sample of 50k, grid by city and color based on weekend
g = sns.FacetGrid(df_train.sample(50000), col="City", 
                  col_wrap=2, height=5, aspect=1.5, hue='Weekend')

#map two first pca's
g.map(sns.scatterplot, "Target_PCA0", "Target_PCA1", alpha=.5 ).add_legend();
g.set_titles('{col_name}', fontsize=17)
plt.show()

# K Means clusterization

#sum of squared distances
ssd = []

K = range(1,5)

#try kmeans for 1:10 clusters - update: shortened to 1:5

#fit and append the inertia_

#inertia_ = sum of squared distances of samples to their closest cluster center

for k in K:
    km = KMeans(n_clusters=k, random_state=4)
    km = km.fit(df_train[min_max_cols])
    ssd.append(km.inertia_)
    
plt.plot(K, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')

plt.show()


km = KMeans(n_clusters=4, random_state=4)
km = km.fit(df_train[min_max_cols])
df_train['clusters_T'] = km.predict(df_train[min_max_cols])


# Plotting clusters

#cross tabulation of city and clusters_T, the prediction of km model

#By default computes a frequency table of the factors unless an array of values 
#and an aggregation function are passed.

#unstack the multiindex to a crosstable

#obtain percentage of every city in a cluster
tmp = pd.crosstab(df_train['City'], df_train['clusters_T'], 
                  normalize='columns').unstack('City').reset_index().rename(columns={0:"perc"})

total = len(df_train)
plt.figure(figsize=(15,16))

plt.subplot(311)

#count observations in every cluster

g = sns.countplot(x="clusters_T", data=df_train)
g.set_title("Cluster Target Count Distribution", fontsize=20)
g.set_ylabel("Count",fontsize= 17)
g.set_xlabel("Target Cluster Distributions", fontsize=17)
sizes=[]
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=14) 
g.set_ylim(0, max(sizes) * 1.15)

#count observations in every cluster, per city
plt.subplot(312)
g1 = sns.countplot(x="clusters_T", data=df_train, hue='City')
g1.set_title("CITIES - Cluster Target Distribution", fontsize=20)
g1.set_ylabel("Count",fontsize= 17)
g1.set_xlabel("Target Cluster Distributions", fontsize=17)
sizes=[]
for p in g1.patches:
    height = p.get_height()
    sizes.append(height)
    g1.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=10)
g1.set_ylim(0, max(sizes) * 1.15)

# box plot
plt.subplot(313)

#values of first Principal component by cluster
g1 = sns.boxplot(x="clusters_T", y='Target_PCA0', 
                 data=df_train, hue='City')
g1.set_title("PCA Feature - Distribution of PCA by Clusters and Cities", 
             fontsize=20)
g1.set_ylabel("PCA 0 Values",fontsize= 17)
g1.set_xlabel("Target Cluster Distributions", fontsize=17)

plt.subplots_adjust(hspace = 0.5)

plt.show()


#PCA values by clusters

#scatterplot PCA0 and PCA1, colored by cluster

plt.figure(figsize=(15,6))

sns.scatterplot(x='Target_PCA0', y='Target_PCA1',
                hue='clusters_T', data=df_train,
                palette='Set1')
plt.title("PCA 0 and PCA 1 by Clusters", fontsize=22)
plt.ylabel("Target PCA 1 values", fontsize=18)
plt.xlabel("Target PCA 0 values", fontsize=18)

plt.show()



# PCA Dispersion by clusters and by city

#sample, colored by cluster, grid by city
g = sns.FacetGrid(df_train.sample(500000), col="City", 
                  col_wrap=2, height=4, aspect=1.5, 
                  hue='clusters_T')

#X and Y of PCAs
g.map(sns.scatterplot, "Target_PCA0", "Target_PCA1", 
      alpha=.5).add_legend();
g.set_titles('{col_name}', fontsize=50)

plt.suptitle("CITIES \nPrincipal Component Analysis Dispersion by Cluster", fontsize=22)

plt.subplots_adjust(hspace = 0.3, top=.85)

plt.show()


# Clusters by hours


#%%

