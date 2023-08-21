import os
import click
import pytest
from click.testing import CliRunner
from surviver_predictor import SurviverPredictor
from cli import cli  

@pytest.fixture
def runner():
    return CliRunner()

def test_train_command(runner):
    result = runner.invoke(cli, ["train", "--model-option", "logistic", "--data-path", "./inputs/train.csv"])
    assert result.exit_code == 0

def test_train_all_models_command(runner):
    result = runner.invoke(cli, ["train_all_models", "--data-path", "./inputs/train.csv"])
    assert result.exit_code == 0

def test_predict_command(runner):
    result = runner.invoke(cli, ["predict", "--model-option", "logistic", "--test-data-path", "./inputs/test.csv"])
    assert result.exit_code == 0