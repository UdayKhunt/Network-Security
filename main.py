from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig , DataValidationConfig , DataTransformationConfig , ModelTrainerConfig

if __name__=='__main__':
    training_pipeline_config = TrainingPipelineConfig()

    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)

    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    print(data_ingestion_artifact)

    print("\n\n\n")
    
    data_validation_config = DataValidationConfig(training_pipeline_config)
    data_validation = DataValidation(data_validation_config , data_ingestion_artifact)
    data_validation_artifact = data_validation.initiate_data_validation()

    print(data_validation_artifact)

    print("\n\n\n")


    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    data_transformation = DataTransformation(data_transformation_config , data_validation_artifact)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    print(data_transformation_artifact)
    
    
    print("\n\n\n")

    model_trainer_config = ModelTrainerConfig(training_pipeline_config)
    model_trainer = ModelTrainer(model_trainer_config , data_transformation_artifact)
    model_trainer_artifact = model_trainer.initiate_model_trainer()

    print(model_trainer_artifact)

