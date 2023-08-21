from data.preprocessing import DataPreprocessor
from model.trainer import ModelTrainer
import pickle

class SurviverPredictor:
    def __init__(self, model_option , training_data='./inputs/train.csv'):
        self.data_path = training_data
        self.model_options = ['logistic', 'random_forest', 'xgboost']
        assert model_option in self.model_options,  print(f'Sorry {model_option} not in {self.model_options}')  
        self. model_option = model_option
        self.data_prepocessor = DataPreprocessor(self.data_path)
        self.df_train = self.data_prepocessor.load_data()
        self.X = self.data_prepocessor.get_x_matrix_from_dataframe(self.df_train)
        self.y = self.df_train['Survived']
        self.trainer = ModelTrainer(self.X, self.y)
        #Load current trained model using the model option
        self.trained_models = {
            'logistic':'./outputs/logistic_model.pkl',
            'random_forest':'./outputs/random_forest_model.pkl',
            'xgboost':'./outputs/xgboost_model.pkl'
        }
        # load the model from the file

        with open(self.trained_models[self.model_option], 'rb') as file:
            self.loaded_model = pickle.load(file)

        # Now you can use the loaded_model for predictions
    def load_last_checkpoint(self):
        # load the model from the file
        checkpoint = self.trained_models[self.model_option]
        with open(checkpoint, 'rb') as file:
            self.loaded_model = pickle.load(file)
        # Now you can use the loaded_model for predictions
        print(f'Loaded Model from checkpoint: {checkpoint}')
        return

    def train(self):
        if self.model_option =='logistic': 
            _ = self.trainer.train_logistic_regression()
        elif self.model_option == 'random_forest':
            _ = self.trainer.train_random_forest()
        else:
            _ = self.trainer.train_xgboost()
        # The model is already saved, load it
        self.load_last_checkpoint()

    def predict(self, test_path_data):
        data_prepocessor = DataPreprocessor(test_path_data)
        df_test = data_prepocessor.load_data()
        X_test = data_prepocessor.get_x_matrix_from_dataframe(df_test)
        print(f'Shape of testing dataset: {df_test.shape}')
        predictions = self.loaded_model.predict(X_test)
        df_test['Predictions'] = predictions
        return df_test
