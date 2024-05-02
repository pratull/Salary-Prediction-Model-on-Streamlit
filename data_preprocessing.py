import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def handle_missing_values(data):
    data.fillna(method='ffill', inplace=True)
    return data

def encode_categorical_variables(data):
    data['Education'] = data['Education'].astype('category').cat.codes
    data['Location'] = data['Location'].astype('category').cat.codes
    data['Job_Title'] = data['Job_Title'].astype('category').cat.codes
    data['Gender'] = data['Gender'].astype('category').cat.codes
    return data

def preprocess_data(file_path):
    data = load_data(file_path)
    data = handle_missing_values(data)
    data = encode_categorical_variables(data)
    X = np.column_stack((data['Education'], data['Experience'], data['Location'], data['Job_Title'], data['Age'], data['Gender']))
    y = data['Salary']
    return X, y
