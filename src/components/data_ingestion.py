from src.exception import NetworkSecurityException
from src.logger import logging

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

import os,sys,pymongo
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv('MONGODB_URL')

class DataIngestion:
    def __init__(self , data_ingestion_config : DataIngestionConfig):
        
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if '_id' in df.columns.to_list():
                df.drop('_id' , axis=1,inplace=True)
            df.replace({'na' : np.nan} , inplace = True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_data_into_feature_store(self , df : pd.DataFrame):
        try:
            feature_store_dir = self.data_ingestion_config.feature_store_dir
            os.makedirs(os.path.dirname(feature_store_dir) , exist_ok=True)
            df.to_csv(feature_store_dir , index = False , header = True)
    
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def split_data_as_train_test(self , df : pd.DataFrame):
        try:
            train_data , test_data = train_test_split(df , test_size = self.data_ingestion_config.train_test_split_ratio)

            dirpath_for_train_test = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dirpath_for_train_test , exist_ok=True)

            train_data.to_csv(self.data_ingestion_config.training_file_path , index = False , header = True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path , index = False , header = True)

        except Exception as e:
            raise NetworkSecurityException(e,sys)

        
    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)

            data_ingestion_artifact = DataIngestionArtifact(self.data_ingestion_config.training_file_path, 
                                                            self.data_ingestion_config.testing_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)