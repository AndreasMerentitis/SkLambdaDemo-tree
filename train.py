try:
  import unzip_requirements
except ImportError:
  pass

import os
import json
import pickle
import time
import functools
import tarfile

import boto3

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

import census_data

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

def trainHandler(event, context):
    time_start = time.time()

    #body = json.loads(event.get('body'))

    # Read in epoch
    #epoch_files = body['epoch']
    epoch_files = ''
    
    logging.warning('first path is %s', os.path.join(epoch_files,census_data.TRAINING_FILE))
    
    logging.warning('second path is %s', FILE_DIR+census_data.TRAINING_FILE)

    # Download files from S3
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,census_data.TRAINING_FILE),
            FILE_DIR+census_data.TRAINING_FILE)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,census_data.EVAL_FILE),
            FILE_DIR+census_data.EVAL_FILE)

    # Setup estimator              
    clf = DecisionTreeClassifier(random_state=0)

    train_Bunch = census_data.load_data_train()
    logging.warning('train_Bunch is %s', train_Bunch)

    clf.fit(train_Bunch.data, train_Bunch.target_numeric.astype(int))
                      
    test_Bunch = census_data.load_data_test(train_Bunch.encoder)
    logging.warning('test_Bunch is %s', test_Bunch)

    predictions = clf.predict(test_Bunch.data)
    
    result = classification_report(test_Bunch.target_numeric.astype(int), predictions, target_names=['low income', 'high income'])
    logging.warning('Evaluation result is %s', result)

    # Zip up model files and store in s3
    with open('/tmp/classifier.pickle', 'wb') as f:
        pickle.dump(clf, f)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).Object(os.path.join(epoch_files,'classifier.pickle')
        ).upload_file(FILE_DIR+'classifier.pickle')

    response = {
        "statusCode": 200,
        "body": json.dumps({'start time': time_start,
                            'runtime': round(time.time()-time_start, 1),
                            'result': result})
    }

    return response
