from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self, X, y, test_size = .2):
        self.X = X
        self.y = y
        self.test_size = test_size

    def train_random_forest(self):

        print('\n#################### Training Random Forest ####################')
        X = self.X
        y = self.y

        # Split the data into training and testing sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize the RandomForestClassifier
        model = RandomForestClassifier(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

        # Perform grid search on the training data
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print('\nBest Parameters:', best_params)

        # Evaluate the model with best parameters on the test data
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        report = classification_report(y_val, y_pred)
        print(f'\nAccuracy with Best Parameters: {accuracy:.5f}')
        print(f'\nF1 with Best Parameters: {f1:.5f}\n')
        print(report)
        # print(best_model)

        # Save the trained model to a file
        model_filename = './outputs/random_forest_model.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)

        print('\n#################### Training Finished ####################')
        return best_model

    def train_xgboost(self):
        print('\n#################### Training XGBoost ####################')
        X = self.X
        y = self.y

        # Split the data into training and testing sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Initialize the XGBClassifier
        model = xgb.XGBClassifier(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

        # Perform grid search on the training data
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print('Best Parameters:', best_params)

        # Evaluate the model with best parameters on the test data
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        report = classification_report(y_val, y_pred)
        print(f'\nAccuracy with Best Parameters: {accuracy:.5f}')
        print(f'\nF1 with Best Parameters: {f1:.5f}\n')
        print(report)
        # print(best_model)

         # Save the trained model to a file
        model_filename = './outputs/xgboost_model.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)
        print('\n#################### Training Finished ####################')
        return best_model
    
    def train_logistic_regression(self):
        print('\n#################### Training Logistic Regression ####################')
        X = self.X
        y = self.y

        # Split the data into training and testing sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Define the parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # Initialize the LogisticRegression model
        model = LogisticRegression(random_state=42)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

        # Perform grid search on the training data
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print('\nBest Parameters:', best_params)

        # Evaluate the model with best parameters on the test data
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        report = classification_report(y_val, y_pred)
        print(f'\nAccuracy with Best Parameters: {accuracy:.5f}')
        print(f'\nF1 with Best Parameters: {f1:.5f}\n')
        print(report)
        # print(best_model)

        # Save the trained model to a file
        model_filename = './outputs/logistic_model.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)
        print('\n#################### Training Finished ####################')
        return best_model