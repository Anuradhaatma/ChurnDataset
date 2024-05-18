import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


class DataCleaning:
    def cleanData(data):
        print(data['TotalCharges'].dtype)
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'],errors = 'coerce')
        print(data['TotalCharges'].dtype)
        data.skew(numeric_only= True)
        print(data.corr(numeric_only= True))

class DataAnalysis:
    def analyseData(data,categorical,numerical,target):
        data[numerical].describe()
        data[numerical].hist(bins=30, figsize=(10, 7))
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        data[data.Churn == "No"][numerical].hist(bins=30, color="blue", alpha=0.5, ax=ax)
        data[data.Churn == "Yes"][numerical].hist(bins=30, color="red", alpha=0.5, ax=ax)
        plt.show()

        # ROWS, COLS = 4, 4
        # fig, ax = plt.subplots(ROWS,COLS, figsize=(19,19))
        # row, col = 0, 0,
        # for i, categorical_feature in enumerate(categorical):
        #     if col == COLS - 1:
        #         row += 1
        # col = i % COLS
        # data[categorical].value_counts().plot(kind='bar', ax=ax[row, col]).set_title(categorical)

        feature = 'Contract'
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, categorical in enumerate(categorical):
            data[data.Churn == "No"][feature].value_counts().plot(kind='bar', ax=ax[0]).set_title('not churned')
            data[data.Churn == "Yes"][feature].value_counts().plot(kind='bar', ax=ax[1]).set_title('churned')
            #plt.show()
    
        data.drop(['customerID'],axis = 1,inplace = True)
        return data