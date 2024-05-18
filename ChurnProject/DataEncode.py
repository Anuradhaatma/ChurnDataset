import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class EncodeData:
    def oneHotEncoder(rowData):
        newData=pd.get_dummies(data=rowData,columns=['gender', 'Partner', 'Dependents', 
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True)
        print(newData.shape)
        newData = newData[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'gender_Male', 'Partner_Yes', 'Dependents_Yes',
       'PhoneService_Yes', 'MultipleLines_No phone service',
       'MultipleLines_Yes', 'InternetService_Fiber optic',
       'InternetService_No', 'OnlineSecurity_No internet service',
       'OnlineSecurity_Yes', 'OnlineBackup_No internet service',
       'OnlineBackup_Yes', 'DeviceProtection_No internet service',
       'DeviceProtection_Yes', 'TechSupport_No internet service',
       'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes',
       'StreamingMovies_No internet service', 'StreamingMovies_Yes',
       'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check','Churn_Yes']]
        return newData

class FeatureScale:
    def imputeAndScale(data):
        data['TotalCharges'] = data['TotalCharges'].replace({None: np.nan})
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        data.TotalCharges = imputer.fit_transform(data["TotalCharges"].values.reshape(-1, 1))
        scaler = StandardScaler()
        scaler.fit(data.drop(['Churn_Yes'],axis = 1))
        scaled_features = scaler.transform(data.drop('Churn_Yes',axis = 1))
        return scaled_features