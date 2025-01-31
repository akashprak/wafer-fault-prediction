from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataclasses import dataclass
from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class modelParams:

    LogisticRegression_params ={
        'penalty':['l1', 'l2'],
        'C':[0.01, 0.1, 1, 10]
    }


    SVC_params = {
        'kernel' : ['linear', 'rbf'],
        'C': [0.1, 0.5, 0.7],
    }


    randomForest_params = {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5],
        'min_samples_split': [3, 5],
        'min_samples_leaf': [2],
        'max_features': ['sqrt', 'log2']
    }


    xgboost_params = {
        'learning_rate': [0.3, 0.2, 0.1],
        'max_depth': [2, 3, 4],
        'n_estimators': [150, 100],
        'colsample_bytree' : [0.5],
        'reg_alpha' : [5],
        'reg_lambda' : [5]
        
    }

class ModelFinder:
    def __init__(self, X_train, y_train) -> None:
        self.params = modelParams()

        self.X_train = X_train
        self.y_train = y_train

    def find_logistic_regression(self) -> dict:
        """
        Returns: dict holding best model and best score
                { "model": best_model
                  "score": best_score }
        """

        try:
            logging.info("Hyperparameter tuning for Logistic Regression:")
            model = LogisticRegression(solver='liblinear', class_weight='balanced')
            model_Grid = GridSearchCV(estimator=model, param_grid=self.params.LogisticRegression_params, 
                                scoring='f1')
            
            model_Grid.fit(self.X_train, self.y_train)
            logging.info(f"Best hyperparameters found for logistic regression:\n{model_Grid.best_params_}\n")
            return {
                "model": model_Grid.best_estimator_,
                "score": model_Grid.best_score_
                }
        
        except Exception as e:
            logging.exception("Hyperparameter tuning failed for Logistic Regression")
            raise CustomException(e)
    
    def find_SVC(self) -> dict:
        """
        Returns: dict holding best model and best score
                { "model": best_model
                  "score": best_score }
        """

        try:
            logging.info("Hyperparameter tuning for SVC:")
            model = SVC(class_weight='balanced')
            model_Grid = GridSearchCV(estimator=model, param_grid=self.params.SVC_params, 
                                scoring='f1', verbose=2)

            model_Grid.fit(self.X_train, self.y_train)
            logging.info(f"Best hyperparameters found for SVC:\n{model_Grid.best_params_}\n")
            return {
                "model": model_Grid.best_estimator_,
                "score": model_Grid.best_score_
                }
        
        except Exception as e:
            logging.exception("Hyperparameter tuning failed for SVC")
            raise CustomException(e)
    
    def find_randomForest(self) -> dict:
        """
        Returns: dict holding best model and best score
                { "model": best_model
                  "score": best_score }
        """

        try:
            logging.info("Hyperparameter tuning for Random Forest:")
            model = RandomForestClassifier(random_state=5, class_weight='balanced')
            model_Grid = GridSearchCV(estimator=model, param_grid=self.params.randomForest_params, 
                                scoring='f1', verbose=2)

            model_Grid.fit(self.X_train, self.y_train)
            logging.info(f"Best hyperparameters found for Random Forest:\n{model_Grid.best_params_}\n")
            return {
                "model": model_Grid.best_estimator_,
                "score": model_Grid.best_score_
                }
        
        except Exception as e:
            logging.exception("Hyperparameter tuning failed for Random Forest")
            raise CustomException(e)
    
    def find_xgboost(self) -> dict:
        """
        Returns: dict holding best model and best score
                { "model": best_model
                  "score": best_score }
        """

        try:
            logging.info("Hyperparameter tuning for XGBoost:")
            model = XGBClassifier(objective="binary:logistic")
            model_Grid = GridSearchCV(estimator=model, param_grid=self.params.xgboost_params, 
                                scoring='f1', verbose=2)

            model_Grid.fit(self.X_train, self.y_train)
            logging.info(f"Best hyperparameters found for XGBoost:\n{model_Grid.best_params_}\n")
            return {
                "model": model_Grid.best_estimator_,
                "score": model_Grid.best_score_
                }
        
        except Exception as e:
            logging.exception("Hyperparameter tuning failed for XGBoost")
            raise CustomException(e)

class ModelTraining:
    """ This class handles the training of different models and picking the best model"""

    def initiate_model_training(self, X_train, y_train):
        try:
            model_finder = ModelFinder(X_train, y_train)
            
            logging.info("Hyperparameter tuning for the models started.")
            logging.info("Scoring metric: f1 score")

            models = {
                "LogisticRegression" : model_finder.find_logistic_regression(),
                "SVC" : model_finder.find_SVC(),
                "RandomForest" : model_finder.find_randomForest(),
                "XGBoost" : model_finder.find_xgboost()
            }
            logging.info("Models trained successfully")
            
            # saving the trained models and finding best model
            best_model_name = None
            best_model = None
            best_score = 0

            for name, model_result in models.items():
                model_score = model_result.get("score")
                model_obj = model_result.get("model")
                logging.info(f"Model name: {name},  f1 score: {model_score}")
                
                model_save_path = Path("artifacts", "Trained_Models", f"{name}.pkl")
                save_object(
                    file_path=model_save_path,
                    obj=model_obj
                )

                if model_score > best_score:
                    best_score = model_score
                    best_model = model_obj
                    best_model_name = name

            print("\n\nBest Model found:")
            print(f"Name: {best_model_name},    f1 score: {best_score}")
            print('='*40, '\n')
            logging.info(f"Best model found,    Model name:{best_model_name}   score:{best_score}")

            best_model_file_path = Path("artifacts", "best_model.pkl")
            save_object(
                file_path=best_model_file_path,
                obj=best_model
            )

        except ValueError:
            logging.exception("Value not found in model_result")
            raise ValueError
        
        except KeyError:
            logging.exception("Incorrect key passed for model_result")
            raise KeyError
        
        except Exception as e:
            logging.exception("initiate_model_training failed")
            raise CustomException(e)