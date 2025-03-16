############ ETL ############
import os,sys,json
from dotenv import load_dotenv
import certifi
from src.exception import NetworkSecurityException
import pandas as pd
import pymongo

load_dotenv()

MONGODB_URL = os.getenv('MONGODB_URL')
ca = certifi.where()

class NetworkDataExtract:
    
    def csv_to_json(self , filepath):
        try:
            df = pd.read_csv(filepath) ##extract
            df.reset_index(drop=True , inplace=True)
            records = list(json.loads(df.T.to_json()).values()) ##transform
            return records
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def load_to_mongo(self , records , database , collection):
        try:
            self.records = records
            self.database = database
            self.collection = collection

            self.mongo_client = pymongo.MongoClient(MONGODB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]

            self.collection.insert_many(records)
            return len(self.records)
    
        except Exception as e:
                raise NetworkSecurityException(e,sys)
        
if __name__ == '__main__':
    obj = NetworkDataExtract()
    records = obj.csv_to_json('dataset/phisingData.csv')
    number_of_records = obj.load_to_mongo(records , 'Uday' , 'NetworkSecurityData')
    print(number_of_records)