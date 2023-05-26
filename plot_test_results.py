"""This script can be used to open, plot, and save the results on the test data."""
import pandas as pd

path = 'ADD_PATH_TO_RESULTS_FOLDER' # Add path to results/model_name/ folder here

df = pd.read_json(path + '/eval.json')
df = df.transpose()
print(df.describe())
df.describe().to_csv(path + '/eval_summary.csv')