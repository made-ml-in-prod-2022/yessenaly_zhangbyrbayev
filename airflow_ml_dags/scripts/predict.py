from argparse import ArgumentParser

import pickle
import pandas as pd
from airflow.models import Variable

best_model_path = Variable.get('best_model')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-od", "--output_dir", 
                        dest="output_dir", 
                        type=str, 
                        default=None, 
                        help="path to directory where to put results",
                        )
    return parser.parse_args()

def predict(output_dir: str):
    with open(best_model_path, 'rb') as file:
        model = pickle.load(file)
    df = pd.read_csv('/opt/airflow/data/processed/data.csv')
    pred = model.predict(df)
    df['prediction'] = pred
    df.to_csv(output_dir + 'predictions.csv', index=False)
    

def main(args):
    predict(output_dir=args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args)
