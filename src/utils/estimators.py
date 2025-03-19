from src.exception import NetworkSecurityException
import sys

class NetworkEstimatior:
    def __init__(self , model , preprocessor):
        try:
            self.model = model
            self.preprocessor = preprocessor
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    def predict(self , X):
        try:
            X = self.preprocessor.transform(X)
            pred = self.model.predict(X)
            return pred
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        