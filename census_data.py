import json
import logging
import os
import pickle

import boto3
import numpy as np
import pandas as pd
import requests
from sklearn.compose import make_column_transformer
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'adult.data'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'adult.test'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)
S3_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']

CENSUS_DATASET = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
)

names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
  ]


def download(data_dir, path=DATA_URL):
  """Download census data if it is not already present."""
  if not os.path.exists(data_dir):
     os.mkdir(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not os.path.exists(training_file_path):
    response = requests.get(TRAINING_URL)
    name = os.path.basename(TRAINING_URL)
    with open(os.path.join(data_dir, name), 'wb') as f:
        f.write(response.content)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not os.path.exists(eval_file_path):
    response = requests.get(EVAL_URL)
    name = os.path.basename(EVAL_URL)
    with open(os.path.join(data_dir, name), 'wb') as f:
        f.write(response.content)
    
 
 
def load_data_train():
    # Load the training and test data, skipping the bad row in the test data
    train = pd.read_csv('/tmp/adult.data', names=names)
    #train = np.loadtxt('/tmp/adult.data', delimiter=',', skiprows=1)
    #train = np.genfromtxt('/tmp/adult.data', delimiter=',', dtype=None, missing_values='?', encoding=None, filling_values=0.0)

    logging.warning('train is %s', train)
    logging.warning('train shape is %s', train.shape)
    logging.warning('train types is %s', train.dtypes)
    
    train['income'] = train['income'].astype('str') 

    data = train.iloc[:, :-1]
    target = train.iloc[:, -1]
    
    target_numeric = target.copy()
    idx_low = train['income'] == ' <=50K'
    idx_high = train['income'] == ' >50K'
    target_numeric.loc[idx_low] = 0
    target_numeric.loc[idx_high] = 1
    
    logging.warning('target_numeric is %s', target_numeric)
    
    #column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
    #                                    ['occupation', 'relationship', 'marital-status', 'race', 'sex', 'native-country']),
    #                                  (OrdinalEncoder(), ['workclass', 'education']),
    #                                  remainder='passthrough')
                                      
    column_trans = OneHotEncoder(handle_unknown='ignore')                                
                                      
    data = column_trans.fit_transform(data)
    logging.warning('data encoded is %s', data)
    logging.warning('data encoded shape is %s', data.shape)
    
    epoch_files = ''
    
    # Zip up preprocess model and store in s3 (to use for inference)
    with open('/tmp/preprocess.pickle', 'wb') as f:
        pickle.dump(column_trans, f)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).Object(os.path.join(epoch_files,'preprocess.pickle')
        ).upload_file(FILE_DIR+'preprocess.pickle')
    
    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        data = data,
        target_numeric = target_numeric,
        target_names = ['income'],
        feature_names = names,
        encoder = column_trans
        #categorical_features = meta['categorical_features'],
    )  
 
 
def load_data_test(encoder):
    # Load the training and test data, skipping the bad row in the test data
    test = pd.read_csv('/tmp/adult.test', names=names, skiprows=1)
   
    logging.warning('test is %s', test)
    logging.warning('test.shape is %s', test.shape)
    
    test['income'] = test['income'].astype('str') 

    data = test.iloc[:, :-1]
    target = test.iloc[:, -1]
    
    target_numeric = target.copy()
    idx_low = test['income'] == ' <=50K.'
    idx_high = test['income'] == ' >50K.'
    target_numeric.loc[idx_low] = 0
    target_numeric.loc[idx_high] = 1
    
    logging.warning('target_numeric is %s', target_numeric)
    
    data1 = encoder.transform(data.head(1))
    logging.warning('data1 point is %s', data1)
    
    logging.warning('data columns is %s', data.columns)
    logging.warning('data types is %s', data.dtypes)
    logging.warning('data raw is %s', data)
    logging.warning('data raw age is %s', data.age)
    logging.warning('data raw workclass is %s', data.workclass)
    data = encoder.transform(data)
    logging.warning('data encoded is %s', data)
        
    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        data = data,
        target_numeric = target_numeric,
        target_names = ['income'],
        feature_names = names,
        encoder = encoder
        #categorical_features = meta['categorical_features'],
    )  
 
 
    
    
    
