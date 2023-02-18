try:
  import unzip_requirements
except ImportError:
  pass

import os
import json
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

    # Create feature columns
    wide_cols, deep_cols = census_data.build_model_columns()
    
    logging.warning('wide_cols are %s', wide_cols)
    logging.warning('deep_cols are %s', deep_cols)

    # Setup estimator
    #classifier = tf.estimator.LinearClassifier(
    #                    feature_columns=wide_cols,
    #                    model_dir=FILE_DIR+'model_'+epoch_files+'/')
                        
    # Setup estimator              
    classifier = DecisionTreeClassifier(random_state=0)

    # Create callable input function and execute train
    train_inpf = functools.partial(
                    census_data.input_fn,
                    FILE_DIR+census_data.TRAINING_FILE,
                    num_epochs=2, shuffle=True,
                    batch_size=64)

    classifier.fit(train_inpf)

    # Create callable input function and execute evaluation
    test_inpf = functools.partial(
                    census_data.input_fn,
                    FILE_DIR+census_data.EVAL_FILE,
                    num_epochs=1, shuffle=False,
                    batch_size=64)

    result = classifier.predict(test_inpf)
    print('Evaluation result: %s' % result)

    # Zip up model files and store in s3
    with open('/tmp/classifier.pickle', 'wb') as f:
        pickle.dump(clf, f)

    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).Object(os.path.join(epoch_files,'classifier.pickle')
        ).upload_file(FILE_DIR+'classifier.pickle')


    # Convert result from float32 for json serialization
    for key in result:
        result[key] = result[key].item()

    response = {
        "statusCode": 200,
        "body": json.dumps({'start time': time_start,
                            'runtime': round(time.time()-time_start, 1),
                            'result': result})
    }

    return response
