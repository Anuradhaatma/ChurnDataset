import LoadData as loader;
import DataCleaning as cleaner;
import DataEncode as encode
import ModelPredictions as model
import pandas as pd

data = loader.loadData()
cleaner.DataCleaning.cleanData(data)

categorical_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
target = "Churn"
data = cleaner.DataAnalysis.analyseData(data,categorical_features,numerical_features,target)
encodeddata= encode.EncodeData.oneHotEncoder(data) 
print (encodeddata.shape)
scaled_features =encode.FeatureScale.imputeAndScale(encodeddata)
logaccuracy = model.LogisticRegressionModel.logisticPrediction(encodeddata,scaled_features)
print(logaccuracy)
svcaccuracy=model.SupportVectorModel.supportVectorPrediction(encodeddata,scaled_features)
print(svcaccuracy)
decisionaccuracy = model.DecisionTreeModel.decisionTreePrediction(encodeddata,scaled_features)
print(decisionaccuracy)
# gridsearchaccuracy = model.GridSearchModel.gridSearchPrediction(endodeddata,scaled_features)
# print(gridsearchaccuracy)
randomforestaccuracy = model.RandomForestModel.randomForestPrediction(encodeddata,scaled_features)
print(randomforestaccuracy)

gradiantaccuracy = model.GradientBoostModel.gradientBoostPrediction(encodeddata, scaled_features)
print(gradiantaccuracy)

summary_data = [['Logistic Regression',logaccuracy],['SVC',svcaccuracy],['DecissionTree', decisionaccuracy]
                ,['Random Forest',randomforestaccuracy], ['Gradiant Boost', gradiantaccuracy]]

summary_df = pd.DataFrame(summary_data, columns=['Method', 'Accuracy'])
print(summary_data)