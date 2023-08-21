import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

    # Get final X Matrix to train and cross validate the results
    def get_x_matrix_from_dataframe(self, dataframe, category_columns = ['Pclass', 'Sex', 'Embarked']):
        """USE CATEGORIES FEATURES AS ONE HOT ENCODER AND TRANFORM SOME CONTINIOUS VARIABLES"""
        #fill nas
        dataframe['Age'] = dataframe['Age'].fillna(dataframe['Age'].mean())
        dataframe['Embarked'] = dataframe['Embarked'].fillna(dataframe['Embarked'].mode())

        temp = pd.get_dummies(dataframe, columns=category_columns)
        temp['log_fare'] = dataframe['Fare'].apply(lambda x: np.log(x+1)).fillna(np.log(1))
        
        X = temp[['Age','SibSp','Parch','log_fare', 
                    "Pclass_1",	"Pclass_2",	"Pclass_3",	"Sex_female",	"Sex_male",	"Embarked_C",	"Embarked_Q",	"Embarked_S"
                    ]]
        return X