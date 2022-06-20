import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-dd", "--data_dir", 
                        dest="data_dir", 
                        type=str, 
                        default=None, 
                        help="path to data directory",
                        )
    parser.add_argument("-od", "--output_dir", 
                        dest="output_dir", 
                        type=str, 
                        default=None, 
                        help="path to output directory",
                        )
    return parser.parse_args()

def preprocess(data_dir: str, output_dir: str):
    df = pd.read_csv(data_dir + 'data.csv')
    target = pd.read_csv(data_dir + 'target.csv')
    df.to_csv(output_dir + 'data.csv', index=False)
    target.to_csv(output_dir + 'target.csv', index=False)

def main(args):
    preprocess(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    # with open('/opt/airflow/data/processed/a.txt', 'w') as file:
    #     file.write('sdfbsrtr')
    args = parse_args()
    main(args)
