import pandas as pd

def loadData():
    df = pd.read_csv('Telco_Customer_Churn.csv')
    df.shape
    print(df.shape)
    df.head()
    df.dtypes
    print(df.columns)
    print(df.info())
    print(df.isnull().sum())
    return df