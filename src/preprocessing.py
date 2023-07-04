# Import libraries
import pandas as pd
import re
from sklearn. preprocessing import StandardScaler
import os

# Import Dataset
df = pd.read_csv(os.getcwd() + '\data\dataset.csv')

"""
Data Preparation Steps
"""

# Convert to datetime
df = df.loc[~(df['CustomerDOB'] == '1/1/1800')]
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], format = '%d/%m/%y')
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format = '%d/%m/%y')

# Creating first and last transaction columns
df['TransactionDateFirst'] = df['TransactionDate'] # to calculate the minimum (first transaction)
df['TransactionDateLast'] = df['TransactionDate'] # to calculate the maximum (last transaction)
df['CustomerAge'] = df['TransactionDate'].dt.year - df['CustomerDOB'].dt.year

# Creating MRF dataframe
MRF_df = df.groupby("CustomerID").agg({
    "TransactionID" : "count",
    "CustGender" : "first",
    "CustLocation":"first",
    "CustAccountBalance"  : "mean",
    "TransactionAmount (INR)" : "mean",
    "CustomerAge" : "median",
    "TransactionDateFirst":"min",
    "TransactionDateLast":"max",
    "TransactionDate":"median"
    })
MRF_df = MRF_df.reset_index()

# Creating frequency and recency columns
MRF_df.rename(columns={"TransactionID":"Frequency"},inplace=True)

MRF_df['Recency']=MRF_df['TransactionDateLast']-MRF_df['TransactionDateFirst']
MRF_df['Recency']=MRF_df['Recency'].astype(str)
MRF_df['Recency']=MRF_df['Recency'].apply(lambda x :re.search('\d+',x).group())
MRF_df['Recency']=MRF_df['Recency'].astype(int)
def rep_0(i):
    if i==0:
        return 1 # 0 days mean that a customer has done transaction recently one time by logic so I will convert 0 to 1
    else:
        return i
MRF_df['Recency']=MRF_df['Recency'].apply(rep_0)

# Dropping unnecessary columns
MRF_df.drop(columns=["TransactionDateFirst",
                     "TransactionDateLast",
                     "CustomerID",
                      "CustLocation",
                      "TransactionDate",
                      ],
                     inplace=True)
MRF_df = MRF_df.reset_index(drop=True)

"""
Data Preprocessing Steps
"""

# Handling missing values
MRF_df["CustGender"].fillna(MRF_df["CustGender"].mode()[0], inplace=True)
MRF_df["CustomerAge"].fillna(MRF_df["CustomerAge"].median(), inplace=True)
MRF_df["CustAccountBalance"].fillna(MRF_df["CustAccountBalance"].median(), inplace=True)

# Encode categorical data
MRF_df['CustGender']=MRF_df['CustGender'].map({'M':1,'F':0})

# Handling negative values
def remove_negative_values(dataframe):
    # Mengambil semua kolom dalam DataFrame
    columns = dataframe.columns
    
    # Menghapus nilai negatif dalam setiap kolom
    for column in columns:
        dataframe[column] = dataframe[column].apply(lambda x: max(x, 0))
    
    return dataframe
MRF_df = remove_negative_values(MRF_df)

# data scaling
df_scaled = StandardScaler().fit_transform(MRF_df)
df_scaled=pd.DataFrame(df_scaled,columns=MRF_df.columns)
df_scaled.head()

# remap gender column
MRF_df['CustGender'] = MRF_df['CustGender'].map({1:'M',0:'F'})

"""
DF to CSV
"""
# Create a CSV file
MRF_df.to_csv(os.getcwd() + '\data\cleaned_data.csv', index=False)
df_scaled.to_csv(os.getcwd() + '\data\standardized_data.csv', index=False)