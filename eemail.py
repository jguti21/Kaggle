# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:59:00 2020

@author: gutierj
"""

# Enron email dataset


#https://www.kaggle.com/zichen/explore-enron


import os
import pandas as pd
import email

import matplotlib.pyplot as plt
%matplotlib inline

import wordcloud
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

import seaborn as sns; sns.set_style('whitegrid')

from nltk.tokenize.regexp import RegexpTokenizer

# Network analysis
import networkx as nx



os.chdir('C:/Users/gutierj/Desktop/Programming/Kaggle/Email')


chunk = pd.read_csv("emails.csv", chunksize = 500)

data = next(chunk)

#data = pd.read_csv("emails.csv")

data.head()

# A single message looks like this
print(data['message'][0])


# Parse the emails into a list email objects
messages = list(map(email.message_from_string, data['message']))


data.drop('message', axis = 1, inplace = True)


# Get fields from parsed email objects
keys = messages[0].keys()


for key in keys:
    data[key] = [doc[key] for doc in messages]




## Helper functions
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)



# Parse content from emails
data['content'] = list(map(get_text_from_email, messages))


print(data['content'][53])



def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs



# Split multiple email addresses
data['From'] = data['From'].map(split_email_addresses)
data['To'] = data['To'].map(split_email_addresses)


# Extract the root of 'file' as 'user'
data['user'] = data['file'].map(lambda x:x.split('/')[0])
del messages

data.head()


for col in data.columns:
    print(col, data[col].nunique())
    
    
# Set index and drop columns with two few values
emails_df = data.set_index('Message-ID')\
    .drop(['file', 'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding'], axis=1)
# Parse datetime
emails_df['Date'] = pd.to_datetime(emails_df['Date'], infer_datetime_format=True)
emails_df.dtypes



del data


#e-mail per year
ax = emails_df.groupby(emails_df['Date'].dt.year)['content'].count().plot()
ax.set_xlabel('Year', fontsize=18)
ax.set_ylabel('N emails', fontsize=18)


#e-mail per day of week
ax = emails_df.groupby(emails_df['Date'].dt.dayofweek)['content'].count().plot()
ax.set_xlabel('Day of week', fontsize=18)
ax.set_ylabel('N emails', fontsize=18)


#e-mail per hour
ax = emails_df.groupby(emails_df['Date'].dt.hour)['content'].count().plot()
ax.set_xlabel('Hour', fontsize=18)
ax.set_ylabel('N emails', fontsize=18)


# Count words in Subjects and content
tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')
emails_df['subject_wc'] = emails_df['Subject'].map(lambda x: len(tokenizer.tokenize(x)))
emails_df['content_wc'] = emails_df['content'].map(lambda x: len(tokenizer.tokenize(x)))


grouped_by_people = emails_df.groupby('user').agg({
        'content': 'count', 
        'subject_wc': 'mean',
        'content_wc': 'mean',
    })
    
    
grouped_by_people.rename(columns={'content': 'N emails', 
                                  'subject_wc': 'Subject word count', 
                                  'content_wc': 'Content word count'}, inplace=True)

grouped_by_people.sort_values('N emails', ascending=False).head()



sns.pairplot(grouped_by_people.reset_index(), hue='user')


# Social network analyses of email senders and recipients

sub_df = emails_df[['From', 'To', 'Date']].dropna()

# drop emails sending to multiple addresses
sub_df = sub_df.loc[sub_df['To'].map(len) == 1]


sub_df = sub_df.groupby(['From', 'To']).count().reset_index()


# Unpack frozensets
sub_df['From'] = sub_df['From'].map(lambda x: next(iter(x)))
sub_df['To'] = sub_df['To'].map(lambda x: next(iter(x)))


# rename column
sub_df.rename(columns={'Date': 'count'}, inplace=True)
sub_df.sort_values('count', ascending=False).head(10)



# Make a network of email sender and receipients
G = nx.from_pandas_edgelist(sub_df, 'From', 'To', edge_attr='count', create_using=nx.DiGraph(),)
print('Number of nodes: %d, Number of edges: %d' % (G.number_of_nodes(), G.number_of_edges()))


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 8))
ax1.hist(list(G.in_degree(weight='count')), log=True, bins=20)
ax1.set_xlabel('In-degrees', fontsize=18)

ax2.hist(list(G.out_degree(weight='count')), log=True, bins=20)
ax2.set_xlabel('Out-degrees', fontsize=18)


#Examine connected components in the network
n_nodes_in_cc = []

for nodes in nx.connected_components(G.to_undirected()):
    n_nodes_in_cc.append(len(nodes))

plt.hist(n_nodes_in_cc, bins=20, log=True)
plt.xlabel('# Nodes in connected components', fontsize=18)
plt.ylim([.1,1e4])



#Wordcloud of subjects
subjects = ' '.join(emails_df['Subject'])
fig, ax = plt.subplots(figsize=(16, 12))
wc = wordcloud.WordCloud(width=800, 
                         height=600, 
                         max_words=200,
                         stopwords=ENGLISH_STOP_WORDS).generate(subjects)
ax.imshow(wc)
ax.axis("off")


#wordcloud of contents
contents = ' '.join(emails_df.sample(1000, replace = True)['content'])
fig, ax = plt.subplots(figsize=(16, 12))
wc = wordcloud.WordCloud(width=800, 
                         height=600, 
                         max_words=200,
                         stopwords=ENGLISH_STOP_WORDS).generate(contents)
ax.imshow(wc)
ax.axis("off")


## Draw network


G = nx.from_pandas_edgelist(sub_df, 'From', 'To', edge_attr='count', create_using=nx.DiGraph(),)

G = nx.DiGraph()
G.add_weighted_edges_from([tuple(x) for x in sub_df.values])
G.edges()

nx.draw_networkx(G)

plt.draw(A)
