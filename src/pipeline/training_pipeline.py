from src.components.data_ingestion import DataIngestion
from src.components.data_transormation import DataTransformation
from src.components.model_trainer import ModelTraining

if __name__=="__main__":
    data_ingestion = DataIngestion()
    train_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, y_train = data_transformation.initiate_data_transformation(train_data_path)

    model_training = ModelTraining()
    model_training.initiate_model_training(X_train, y_train)