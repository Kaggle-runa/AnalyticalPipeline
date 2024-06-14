import argparse
from src.usecase.data_preparation import prepare_data
from src.usecase.data_analysis import analyze_data
from src.usecase.model_training import train_and_save_model
from usecase.hyperparameter_tuner import tune_model_hyperparameters

def main():
    parser = argparse.ArgumentParser(description='Data Processing and Model Training Application')
    parser.add_argument('--prepare', action='store_true', help='Prepare data')
    parser.add_argument('--analyze', action='store_true', help='Analyze data')
    parser.add_argument('--train', action='store_true', help='Train and save model')
    parser.add_argument('--tune', action='store_true', help='Tune model hyperparameters')
    parser.add_argument('--processed_file', type=str, help='Specify the processed file to use for training')

    args = parser.parse_args()

    if args.prepare:
        prepare_data()
    if args.analyze:
        analyze_data()
    if args.train:
        train_and_save_model(args.processed_file)
    if args.tune:
        tune_model_hyperparameters()

if __name__ == '__main__':
    main()
