from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score ,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib as joblib

class LogisticRegressionModel:
    def logisticPrediction (encodeddata,scaled_features):
        X = scaled_features
        Y = encodeddata['Churn_Yes']
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)
        logmodel = LogisticRegression()
        logmodel.fit(X_train,Y_train)
        predLR = logmodel.predict(X_test)
        print(classification_report(Y_test, predLR))
        # calculate the classification report
        report = classification_report(Y_test, predLR, target_names=['Churn_No', 'Churn_Yes'])

        # split the report into lines
        lines = report.split('\n')

        # split each line into parts
        parts = [line.split() for line in lines[2:-5]]

        # extract the metrics for each class
        class_metrics = dict()
        for part in parts:
            class_metrics[part[0]] = {'precision': float(part[1]), 'recall': float(part[2]), 'f1-score': float(part[3]), 'support': int(part[4])}

        # create a bar chart for each metric
        fig, ax = plt.subplots(1, 4, figsize=(12, 4))
        metrics = ['precision', 'recall', 'f1-score', 'support']
        for i, metric in enumerate(metrics):
            ax[i].bar(class_metrics.keys(), [class_metrics[key][metric] for key in class_metrics.keys()])
            ax[i].set_title(metric)

        # display the plot
        plt.show()
        confusion_matrix_LR = confusion_matrix(Y_test, predLR)
        
        plt.matshow(confusion_matrix(Y_test, predLR))

        # add labels for the x and y axes
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion_matrix_LR[i, j], ha='center', va='center')


        # Add custom labels for x and y ticks
        plt.xticks([0, 1], ["Not Churned", "Churned"])
        plt.yticks([0, 1], ["Not Churned", "Churned"])
        plt.show()

        print(logmodel.score(X_train, Y_train))
        print(accuracy_score(Y_test, predLR))

        return accuracy_score(Y_test, predLR)

class SupportVectorModel:
    def supportVectorPrediction(encodeddata,scaled_features):
        X = scaled_features
        Y = encodeddata['Churn_Yes']
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)
        svc = SVC()
        svc.fit(X_train, Y_train)
        y_pred_svc = svc.predict(X_test)
        print(classification_report(Y_test, y_pred_svc))
        confusion_matrix_svc = confusion_matrix(Y_test, y_pred_svc)
        # create a heatmap of the matrix using matshow()

        plt.matshow(confusion_matrix_svc)

        # add labels for the x and y axes
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion_matrix_svc[i, j], ha='center', va='center')

        
        # Add custom labels for x and y ticks
        plt.xticks([0, 1], ["Not Churned", "Churned"])
        plt.yticks([0, 1], ["Not Churned", "Churned"])
        plt.show()
        model_filename = 'svc_model.joblib'
        joblib.dump(svc, model_filename)
                    
        svc.score(X_train,Y_train)
        return accuracy_score(Y_test, y_pred_svc)

class DecisionTreeModel:
    def decisionTreePrediction(encodeddata,scaled_features):
        X = scaled_features
        Y = encodeddata['Churn_Yes']
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, Y_train)
        y_pred_dtc = dtc.predict(X_test)
        print(classification_report(Y_test, y_pred_dtc))
        confusion_matrix_dtc = confusion_matrix(Y_test, y_pred_dtc)
        plt.matshow(confusion_matrix_dtc)

        # add labels for the x and y axes
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion_matrix_dtc[i, j], ha='center', va='center')


        # Add custom labels for x and y ticks
        plt.xticks([0, 1], ["Not Churned", "Churned"])
        plt.yticks([0, 1], ["Not Churned", "Churned"])
        plt.show()

        dtc.score(X_train,Y_train)
        return accuracy_score(Y_test, y_pred_dtc)
    
class GridSearchModel:
    def gridSearchPrediction(encodeddata,scaled_features):
        X = scaled_features
        Y = encodeddata['Churn_Yes']
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)
        dtc = DecisionTreeClassifier(random_state= 42)
        param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10,20, 30, 40, 50],
        'min_samples_split': [2,5,10],
        'min_samples_leaf': [1,2,4]
        }
        grid_search = GridSearchCV(estimator= dtc, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, Y_train)
        best_params =grid_search.best_params_
        best_clf= DecisionTreeClassifier(random_state=42, **best_params)
        best_clf.fit(X_train,Y_train)
        y_pred_gs = best_clf.predict(X_test)
        print(classification_report(Y_test, y_pred_gs))
        confusion_matrix_dtc = confusion_matrix(Y_test, y_pred_gs)
        plt.matshow(confusion_matrix_dtc)

        # add labels for the x and y axes
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion_matrix_dtc[i, j], ha='center', va='center')


        # Add custom labels for x and y ticks
        plt.xticks([0, 1], ["Not Churned", "Churned"])
        plt.yticks([0, 1], ["Not Churned", "Churned"])
        plt.show()

        grid_search.score(X_train,Y_train)
        fpr, tpr, _ = metrics.roc_curve(Y_test,  y_pred_gs)

        #create ROC curve
        plt.plot(fpr,tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        return accuracy_score(Y_test, y_pred_gs)
    

class RandomForestModel:
    def randomForestPrediction(encodeddata,scaled_features):
        X = scaled_features
        Y = encodeddata['Churn_Yes']
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, Y_train)
        rf_y_pred = rf_clf.predict(X_test)
        print(classification_report(Y_test, rf_y_pred))
        confusion_matrix_dtc = confusion_matrix(Y_test, rf_y_pred)
        plt.matshow(confusion_matrix_dtc)

        # add labels for the x and y axes
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion_matrix_dtc[i, j], ha='center', va='center')


        # Add custom labels for x and y ticks
        plt.xticks([0, 1], ["Not Churned", "Churned"])
        plt.yticks([0, 1], ["Not Churned", "Churned"])
        plt.show()

        rf_clf.score(X_train,Y_train)
        return accuracy_score(Y_test, rf_y_pred)
    
class GradientBoostModel:
    def gradientBoostPrediction(encodeddata,scaled_features):
        X = scaled_features
        Y = encodeddata['Churn_Yes']
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)
        gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_clf.fit(X_train, Y_train)
        gb_y_pred = gb_clf.predict(X_test)
        print(classification_report(Y_test, gb_y_pred))
        confusion_matrix_dtc = confusion_matrix(Y_test, gb_y_pred)
        plt.matshow(confusion_matrix_dtc)

        # add labels for the x and y axes
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        for i in range(2):
            for j in range(2):
                plt.text(j, i, confusion_matrix_dtc[i, j], ha='center', va='center')


        # Add custom labels for x and y ticks
        plt.xticks([0, 1], ["Not Churned", "Churned"])
        plt.yticks([0, 1], ["Not Churned", "Churned"])
        plt.show()

        gb_clf.score(X_train,Y_train)
        return accuracy_score(Y_test, gb_y_pred)