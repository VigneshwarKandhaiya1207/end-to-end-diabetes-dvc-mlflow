import pathlib
import os
import sys


files_to_be_created=["src/__init__.py",
                     "experiments.ipynb",
                     "params.yaml",
                     "dvc.yaml",
                     "src/preprocess.py",
                     "src/train.py",
                     "src/evaluate.py"
                     ]


for files in files_to_be_created:
    filepath=pathlib.Path(files)
    filedir,filename=os.path.split(filepath)

    print(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True)
    if (not os.path.exists(files)) or (os.path.getsize(files) > 0):
        with open(files,"w"):
            pass

