import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor # type: ignore
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent avriables from train and test data')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                # "Ridge": Ridge(),
                # "Lasso": Lasso(),
                # "SVR": SVR()
            }
            
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                logging.info('Best model has r2 score less than 60%')
                raise CustomException('No Best Model Found')
            
            print(f'Best Model Found, Model Name:{best_model},R2 Score:{best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found,Model Name: {best_model_name},R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for catboost')

            # # Hyperparameter tuning on Catboost
            # # Initializing catboost
            # cbr = CatBoostRegressor(verbose=False)

            # # Creating the hyperparameter grid
            # param_dist = {'depth'          : [4,5,6,7,8,9, 10],
            #               'learning_rate' : [0.01,0.02,0.03,0.04],
            #               'iterations'    : [300,400,500,600]}

            # #Instantiate RandomSearchCV object
            # rscv = RandomizedSearchCV(cbr , param_dist, scoring='r2', cv =5, n_jobs=-1)

            # # Fit the model
            # rscv.fit(xtrain, ytrain)

            # # Print the tuned parameters and score
            # print(f'Best Catboost parameters : {rscv.best_params_}')
            # print(f'Best Catboost Score : {rscv.best_score_}')
            # print('\n====================================================================================\n')

            # best_cbr = rscv.best_estimator_

            # logging.info('Hyperparameter tuning complete for Catboost')

            # logging.info('Hyperparameter tuning started for KNN')

            # # Initialize knn
            # knn = KNeighborsRegressor()

            # # parameters
            # k_range = list(range(2, 31))
            # param_grid = dict(n_neighbors=k_range)

            # # Fitting the cvmodel
            # grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
            # grid.fit(xtrain, ytrain)

            # # Print the tuned parameters and score
            # print(f'Best KNN Parameters : {grid.best_params_}')
            # print(f'Best KNN Score : {grid.best_score_}')
            # print('\n====================================================================================\n')

            # best_knn = grid.best_estimator_

            # logging.info('Hyperparameter tuning Complete for KNN')

            # logging.info('Voting Regressor model training started')

            # # Creating final Voting regressor
            # er = VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
            # er.fit(xtrain, ytrain)
            # print('Final Model Evaluation :\n')
            # print_evaluated_results(xtrain,ytrain,xtest,ytest,er)
            # logging.info('Voting Regressor Training Completed')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj =best_model
            )
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            # mae, rmse, r2 = model_metrics(y_test, predicted) 
            return r2_square   
        except Exception as e:
            # logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

# ... existing code ...

if __name__ == "__main__":
    # Load actual data from artifacts
    import pandas as pd
    import numpy as np
    
    try:
        # Load your actual data
        train_df = pd.read_csv("artifacts/train.csv")
        test_df = pd.read_csv("artifacts/test.csv")
        
        # Prepare the data (assuming math_score is your target)
        target_column = "math_score"
        
        # Prepare training data
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        train_array = np.column_stack([X_train.values, y_train.values])
        
        # Prepare test data
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        test_array = np.column_stack([X_test.values, y_test.values])
        
        # Initialize and run the model trainer
        trainer = ModelTrainer()
        mae, rmse, r2 = trainer.initiate_model_trainer(train_array, test_array)
        print(f"Training completed successfully!")
        print(f"MAE: {mae}, RMSE: {rmse}, R2: {r2}")
        
    except Exception as e:
        print(f"Error during training: {e}")