import os
import yaml
import numpy as np
import pandas as pd



def create_directories(directory_name):
    try:
        if (not os.path.exists(directory_name)):
            os.makedirs(directory_name,exist_ok=True)
            print("Directory created successfully!!!")
        else:
            print("Directory already exists.")
    except Exception as e:
        print("Directory creation failed.")


def load_data(data_path,output_path):
    data=pd.read_csv(data_path)
    data.to_csv(os.path.join(output_path,"data.csv"),index=False)


if __name__=="__main__":
    input_path="https://raw.githubusercontent.com/VigneshwarKandhaiya1207/datasets/refs/heads/main/diabetes.csv"
    output_path=yaml.safe_load(open("params.yaml"))["directories"]["data_path"]
    create_directories(output_path)
    load_data(input_path,output_path)


