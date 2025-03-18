from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path : str

@dataclass
class DataValidationArtifact:
    validation_status : bool
    valid_train_filepath : str
    valid_test_filepath : str
    invalid_train_filepath : str
    invalid_test_filepath : str
    drift_report_file_path : str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path : str
    transformed_test_file_path : str
    preprocessor_obj_path : str

@dataclass
class ClassificationMetricArtifact:
    precision_score:float
    recall_score:float
    f1_score:float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path : str
    train_metric_artifact : ClassificationMetricArtifact
    test_metric_artifact : ClassificationMetricArtifact
