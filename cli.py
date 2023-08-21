import click
from surviver_predictor import SurviverPredictor
import datetime
import time

@click.group()
def cli():
    """Titanic Surviver Predictor Package CLI 🚢"""
    pass


@cli.command()
@click.option('--model-option', required=True, help='Choose one of the different models to use. (logistic, random forest, xgboost)')
@click.option('--data-path', type=click.Path(exists=True), required=True, help='Path to the training dataset CSV file')
def train(model_option, data_path):
    """Train the ML model"""
    t0 = time.time()
    predictor = SurviverPredictor(model_option, data_path)
    print(predictor.loaded_model)
    predictor.train()
    print(predictor.loaded_model)
    # Print or save the trained model as needed
    t1 = time.time()
    print(f'Elapsed Time: {t1-t0:.5f} Seconds')
    return

@cli.command()
@click.option('--data-path', type=click.Path(exists=True), required=True, help='Path to the training dataset CSV file')
def train_all_models(data_path):
    """Train all the Ml Models in the options"""
    t0 = time.time()
    for model_option in SurviverPredictor('logistic').model_options:
        predictor = SurviverPredictor(model_option, data_path)
        predictor.train()
        print(predictor.loaded_model)
    # Print or save the trained model as needed
    t1 = time.time()
    print(f'Elapsed Time: {t1-t0:.5f} Seconds')
    return

@cli.command()
@click.option('--model-option', required=True, help='Choose one of the different models to use. (logistic, random forest, xgboost)')
@click.option('--test-data-path', type=click.Path(exists=True), required=True, help='Path to the test dataset CSV file')
def predict(model_option, test_data_path):
    """Make a prediction using a loaded checkpoint"""
    t0 = time.time()
    print(model_option,test_data_path )
    predictor = SurviverPredictor(model_option)
    # print(predictor.trainer)
    # print(predictor.loaded_model)
    df_test = predictor.predict(test_data_path)
    _today = datetime.datetime.now().strftime('%Y_%m_%d')
    #Export results
    output_path = f'./results/res_{model_option}_{_today}.csv'
    df_test.to_csv(output_path, index=False)
    t1 = time.time()
    print(f'\nPredictions finished and saved them into {output_path}')
    print(f'Elapsed Time: {t1-t0:.5f} Seconds')
    return df_test



if __name__ == '__main__':
    cli()
