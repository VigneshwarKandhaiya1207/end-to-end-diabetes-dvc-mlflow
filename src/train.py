import os
import mlflow
import yaml
import pandas as pd
import numpy as np
import pickle
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
from urllib.parse import urlparse


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/vigneshwar_kandhaiya/end-to-end-diabetes-dvc-mlflow.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="vigneshwar_kandhaiya"
os.environ["MLFLOW_TRACKING_PASSWORD"]="5e52108ecd9bdea4007d036d3ccefc4af246697e"


def create_directories(directory_name):
    try:
        if (not os.path.exists(directory_name)):
            os.makedirs(directory_name,exist_ok=True)
            print("Directory created successfully!!!")
        else:
            print("Directory already exists.")
    except Exception as e:
        print("Directory creation failed.")


def hyperparameter_tuning(X_train,y_train,params):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=params,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train) 
    print("training completed!!!")

    return grid_search


def train(data_path,model_path):
    data=pd.read_csv(os.path.join(data_path,"data.csv"))
    X=data.drop(["Outcome"],axis=1)
    y=data["Outcome"]
    mlflow.set_tracking_uri("https://dagshub.com/vigneshwar_kandhaiya/end-to-end-diabetes-dvc-mlflow.mlflow")
    with mlflow.start_run():
        X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
        signature=infer_signature(X_train,y_train)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        grid_search=hyperparameter_tuning(X_train,y_train,param_grid)
        best_model=grid_search.best_estimator_

        y_pred=best_model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        cm=confusion_matrix(y_test,y_pred)
        cr=classification_report(y_test,y_pred)
        
        best_params={"best_n_estimators":grid_search.best_params_["n_estimators"],
                           "max_depth":grid_search.best_params_["max_depth"],
                           "best_min_samples_split":grid_search.best_params_["min_samples_split"],
                           "best_min_samples_leaf":grid_search.best_params_["min_samples_leaf"]}

        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_params(best_params)
        
        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cr,"classification_report.txt")

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!="file":
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best model")
        else:
            mlflow.sklearn.log_model(best_model,signature=signature)
        
        ## Save the model
        model_name=os.path.join(model_path,"random_forest.pkl")

        with open (model_name,"wb") as file_handler:
            pickle.dump(best_model,file_handler)
        
        print("model saved to {}".format(model_name))

if __name__=="__main__":
    data_path=yaml.safe_load(open("params.yaml"))["directories"]["data_path"]
    model_path=yaml.safe_load(open("params.yaml"))["directories"]["model_path"]
    create_directories(model_path)
    train(data_path=data_path,model_path=model_path)

    


