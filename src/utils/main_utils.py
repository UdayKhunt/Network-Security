from src.exception import NetworkSecurityException
from src.logger import logging
import yaml,sys
import numpy as np
import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(filepath):
    try:
        with open(filepath , 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
            
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def write_yaml_file(filepath , content):
    try:
        with open(filepath , 'w') as yaml_file:
            yaml.dump(content , yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_numpy_array(filepath , array):
    try:
        os.makedirs(os.path.dirname(filepath) , exist_ok=True)
        with open(filepath , 'wb') as arr:
            np.save(arr , array)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def save_obj(filepath , obj):
    try:
        os.makedirs(os.path.dirname(filepath) , exist_ok=True)
        with open(filepath , 'wb') as arr:
            pickle.dump(obj , arr)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def load_numpy_array(filepath):
    try:
        with open(filepath , 'rb') as arr:
            return np.load(arr)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def load_obj(filepath):
    try:
        with open(filepath , 'rb') as obj:
            return pickle.load(obj)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    

def evaluate_models(X_train , Y_train , X_test , Y_test , models , params):
    try:
        report = {}
        for i in range(len(models)):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            param = params[model_name]

            grid = GridSearchCV(model , param , cv = 3)
            grid.fit(X_train , Y_train)

            model.set_params(**grid.best_params_)

            model.fit(X_train , Y_train)
            Y_pred = model.predict(X_test)
            r2 = r2_score(Y_test , Y_pred)

            report[model_name] = r2
        
        return report

    except Exception as e:
        raise NetworkSecurityException(e,sys)