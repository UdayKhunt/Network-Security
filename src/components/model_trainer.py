from src.exception import NetworkSecurityException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact , ModelTrainerArtifact , ClassificationMetricArtifact
from src.utils.main_utils import load_obj , load_numpy_array, evaluate_models, save_obj
from src.utils.classification_metrics import get_classification_metrics

from sklearn.ensemble import AdaBoostClassifier , RandomForestClassifier , GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import sys

class ModelTrainer:
    def __init__(self , model_trainer_config : ModelTrainerConfig, data_transformation_artifact : DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def track_mlflow(self , best_model , classifaction_metric : ClassificationMetricArtifact):
        with mlflow.start_run():
            f1 = classifaction_metric.f1_score
            precision = classifaction_metric.precision_score
            recall = classifaction_metric.recall_score

            mlflow.log_metric('f1_score' , f1)
            mlflow.log_metric('precision' , precision)
            mlflow.log_metric('recall' , recall)

            mlflow.sklearn.log_model(best_model , 'model')
    
    def train_and_evaluate_models(self,X_train , Y_train , X_test , Y_test):
        try:
            models = {
                'Random Forest' : RandomForestClassifier(),
                'Decision Tree' : DecisionTreeClassifier(),
                'Gradient Boosting' : GradientBoostingClassifier(),
                'Logistic Regression' : LogisticRegression(),
                'AdaBoost' : AdaBoostClassifier()
            }

            params={
                "Random Forest":{
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32]
                },
                "Decision Tree": {
                    'criterion':['gini', 'entropy'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                
                "Gradient Boosting":{
                    # 'loss':['log_loss', 'exponential'],
                    'learning_rate':[.1,.01,.05],
                    'subsample':[0.6,0.7,0.75],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32]
                },
                "Logistic Regression":{},
                "AdaBoost":{
                    'learning_rate':[.1,.01,.001],
                    'n_estimators': [8,16,32]
                }
            }

            model_report = evaluate_models(X_train , Y_train , X_test , Y_test , models , params)
            max_score = max(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(max_score)]
            best_model = models[best_model_name]

            best_model.fit(X_train , Y_train)
            Y_train_pred = best_model.predict(X_train)
            classfication_train_metric_artifact = get_classification_metrics(Y_train , Y_train_pred)

            self.track_mlflow(best_model , classfication_train_metric_artifact)

            Y_test_pred = best_model.predict(X_test)
            classification_test_metric_artifact = get_classification_metrics(Y_test , Y_test_pred)
            
            self.track_mlflow(best_model , classification_test_metric_artifact)

            save_obj(self.model_trainer_config.trained_model_file_path , best_model)

            return (self.model_trainer_config.trained_model_file_path,
                    classfication_train_metric_artifact , 
                    classification_test_metric_artifact
                    )

        except Exception as e:
            raise NetworkSecurityException(e,sys)


    def initiate_model_trainer(self):
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            X_train , Y_train , X_test , Y_test = (
                train_arr[:,:-1], train_arr[:,-1],
                test_arr[:,:-1] , test_arr[:,-1]
            )

            model_trainer_artifact =  self.train_and_evaluate_models(X_train , Y_train , X_test , Y_test)
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e,sys)