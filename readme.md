# Titanic-CLI: Titanic - Machine Learning from Disaster 🚢

![Titanic](./giphy.gif)

This repository contains my approach to the Titanic: Machine Learning from Disaster competition on Kaggle. The goal of this competition is to predict which passengers survived the Titanic shipwreck using machine learning techniques. This README provides an overview of the competition, the contents of this repository, and how to get started.

## Competition Overview

The Titanic competition is a classic machine learning problem that challenges participants to predict passenger survival based on various features such as age, sex, class, and more. It's a great introduction to machine learning concepts and techniques for beginners and provides an opportunity to explore data preprocessing, feature engineering, model selection, and evaluation.

For more details about the competition, visit the [Kaggle competition page](https://www.kaggle.com/competitions/titanic/overview).

## Repository Contents

This repository contains the following files and directories:

- `inputs/`: A directory containing the dataset files (train and test CSV files).
- `outputs/`: A list of 3 trained models to use. (Logistic Regression, Random Forest and a XGBoost).
- `data/`: All the code for read and preprocess the data.
- `model/`: All the code for training the different variations of models.
- `notebook/`: A Jupyter notebook where I explore the dataset, preprocess the data, build and evaluate models.
- `results`: A directory to store the results of the models in the test file.
- `README.md`: The README file you're currently reading.
- `cli.py`: The CLI Programming Interface to Interact with the code.

## Getting Started

To get started with this repository:

1. Clone this repository:

   ```bash
   git clone https://github.com/FelipeAce96/titanic-cli
   cd titanic-cli
   ```
2. Set up a virtual environment (recommended).

3. Install the required dependencies. You can do this using `pip`:

   ```bash
   pip install -r requirements.txt

4. Use the CLI interface and his diferrents commands:

   ```bash (predictions)
   python cli.py predict --test-data-path {your_test_path} --model-option {logistic | random_forest | xgboost}
   ```

   ```bash (Train one of the models)
   python cli.py train --data-path {your_training_path} --model-option {logistic | random_forest | xgboost}
   ```

   ```bash (Train all the models)
   python cli.py train-all-models --data-path {your_training_path}
   ```

5. You can rome some test using the following commands:

   ```bash
   coverage run -m pytest test_cli.py
   ```

   ```bash
   coverage report -m
   ```

6. You can run the notebook using google colab is the easiest way. 🚀