import os
import pandas as pd
import email

# import a sample to develop the cleaning
inputrows = 3050
df = pd.read_csv("./data/emails.csv",
                        nrows=inputrows)

# Patter in the message part:
# Name of the var : var

test = df["message"].str.split(":")

def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs

# Parse the emails into a list email objects
messages = list(map(email.message_from_string,
                    df['message']))

df.drop('message',
        axis=1,
        inplace=True)

# Get fields from parsed email objects
keys = messages[0].keys()
for key in keys:
    df[key] = [doc[key] for doc in messages]
# Parse content from emails
df['content'] = list(map(get_text_from_email, messages))
# Split multiple email addresses
df['From'] = df['From'].map(split_email_addresses)
df['To'] = df['To'].map(split_email_addresses)

# Extract the root of 'file' as 'user'
df['user'] = df['file'].map(lambda x: x.split('/')[0])
del messages

# Write the df
df.to_csv("./data/emails_v2.csv")

