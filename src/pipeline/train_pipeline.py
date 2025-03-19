from src.entity.config_entity import TrainingPipelineConfig , DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact , ModelTrainerArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.main_utils import save_obj, load_obj

from src.exception import NetworkSecurityException
from src.logger import logging
import sys

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e , sys)
        
    def start_data_validation(self, data_ingestion_artifact : DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config , data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e , sys)
        
    def start_data_transformation(self , data_validation_artifact : DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config , data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e , sys)
    
    def start_model_trainer(self , data_transformation_artifact : DataTransformationArtifact):
        model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config , data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        return model_trainer_artifact
    

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact : ModelTrainerArtifact = self.start_model_trainer(data_transformation_artifact)
            
            model = load_obj(model_trainer_artifact.trained_model_file_path)
            preprocessor = load_obj(data_transformation_artifact.preprocessor_obj_path)
            save_obj('final_objects/model.pkl' , model)
            save_obj('final_objects/preprocessor.pkl' , preprocessor)
            
        except Exception as e:
            raise NetworkSecurityException(e , sys)
    
if __name__=='__main__':
    train_pipeline = TrainingPipeline()
    train_pipeline.run_pipeline()