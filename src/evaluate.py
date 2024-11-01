import os
import pandas as pd
import pickle
import mlflow
import yaml
from sklearn.metrics import accuracy_score


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/vigneshwar_kandhaiya/end-to-end-diabetes-dvc-mlflow.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="vigneshwar_kandhaiya"
os.environ["MLFLOW_TRACKING_PASSWORD"]="5e52108ecd9bdea4007d036d3ccefc4af246697e"



def evaluate_model(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    mlflow.set_tracking_uri("https://dagshub.com/vigneshwar_kandhaiya/end-to-end-diabetes-dvc-mlflow.mlflow")
    logged_model = 'runs:/a5bb959021374dc19adebb381847bf89/model'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    y_pred_model_hub=loaded_model.predict(X)
    accuracy_model_hub=accuracy_score(y,y_pred_model_hub)

    ## Loading from the file system:

    model=pickle.load(open(model_path,"rb"))
    y_pred_file_path=model.predict(X)
    accuracy_file_path=accuracy_score(y,y_pred_file_path)

    print("The model hub accuracy is : {}".format(accuracy_model_hub))
    print("The file path accuracy is : {}".format(accuracy_file_path))

if __name__ == "__main__":
    data_path=os.path.join(yaml.safe_load(open("params.yaml"))["directories"]["data_path"],"data.csv")
    model_path=os.path.join(yaml.safe_load(open("params.yaml"))["directories"]["model_path"],"random_forest.pkl")
    evaluate_model(data_path,model_path)





