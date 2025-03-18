from src.exception import NetworkSecurityException
from src.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import precision_score , recall_score , f1_score

def get_classification_metrics(Y_test , Y_pred):
    precision = precision_score(Y_test , Y_pred)
    recall = recall_score(Y_test , Y_pred)
    f1 = f1_score(Y_test , Y_pred)

    return ClassificationMetricArtifact(precision , recall , f1)