try:
  import unzip_requirements
except ImportError:
  pass

import json
import os
import pickle
import tarfile
import time

import boto3
import numpy as np
import pandas as pd

import census_data

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']

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
    
def _predict_point(predict_input_point, epoch_files):
    """
    Makes predictions for a signle data point
    """
    
    logging.warning('predict_input_point is %s', predict_input_point)
    
    df_predict_input_point = pd.DataFrame.from_dict(predict_input_point, orient='columns')
    df_predict_input_point.columns = names
    logging.warning('df_predict_input_point shape is %s', df_predict_input_point.shape)
    
    df_predict_input_point = df_predict_input_point.iloc[:, :-1]
    
    df_predict_input_point['age'] = df_predict_input_point['age'].astype('int') 
    df_predict_input_point['fnlwgt'] = df_predict_input_point['fnlwgt'].astype('int') 
    df_predict_input_point['education-num'] = df_predict_input_point['education-num'].astype('int') 
    df_predict_input_point['capital-gain'] = df_predict_input_point['capital-gain'].astype('int') 
    df_predict_input_point['capital-loss'] = df_predict_input_point['capital-loss'].astype('int') 
    df_predict_input_point['hours-per-week'] = df_predict_input_point['hours-per-week'].astype('int') 
    
    logging.warning('data columns is %s', df_predict_input_point.columns)
    logging.warning('data types is %s', df_predict_input_point.dtypes)
    logging.warning('data raw is %s', df_predict_input_point)
    logging.warning('data raw age is %s', df_predict_input_point.age)
    logging.warning('data raw workclass is %s', df_predict_input_point.workclass)
    
    # Download model from S3 and extract
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,'preprocess.pickle'),
            FILE_DIR+'preprocess.pickle')

    # Download model from S3 and extract
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,'classifier.pickle'),
            FILE_DIR+'classifier.pickle')

    # Load preprocess transformation pipeline
    encoder = pickle.load(open(FILE_DIR+'preprocess.pickle', 'rb'))
    logging.warning('encoder is %s', encoder)

    df_predict_input_point_final = encoder.transform(df_predict_input_point)
    logging.warning('df_predict_input_point_final is %s', df_predict_input_point_final)

    # Load model
    clf = pickle.load(open(FILE_DIR+'classifier.pickle', 'rb'))
    logging.warning('clf is %s', clf)

    # Setup prediction
    predictions = clf.predict(df_predict_input_point_final)
    logging.warning('predictions is %s', predictions)
    
    return predictions


def inferHandler(event, context): 
    run_from_queue = False 
    try:
        # This path is executed when the lamda is invoked directly
        body = json.loads(event.get('body'))
    except:
        # This path is executed when the lamda is invoked through the lambda queue
        run_from_queue = True
        body = event

    logging.warning('body is %s', body)
    # Read in prediction data as dictionary
    # Keys should match _CSV_COLUMNS, values should be lists
    predict_input = body['input']
    
    logging.warning('predict_input type is %s', type(predict_input))
    logging.warning('predict_input is %s', predict_input)
    
    # Read in epoch
    epoch_files = body['epoch']
    epoch_files = ''
    
    logging.warning('run_from_queue is %s', run_from_queue)
    
    predictions_batch = []
    if isinstance(predict_input, list) and not run_from_queue: 
        # Direct call with many datapoints
        for jj in range(len(predict_input)):
            predict_input_point = predict_input[jj][0]
            predictions = _predict_point(predict_input_point, epoch_files)
            predictions_batch.append(predictions.tolist())
    elif run_from_queue: 
        # Call from lambda queue
        predict_input_point = predict_input[0]
        if isinstance(predict_input_point, list):
           predict_input_point = predict_input_point[0]
        logging.warning('predict_input_point is %s', predict_input_point)
        predictions = _predict_point(predict_input_point, epoch_files)
        logging.warning('predictions is %s', predictions)
        predictions_batch.append(predictions.tolist())
    else: 
        # Direct call with one datapoint
        predict_input_point = predict_input
        predictions = _predict_point(predict_input_point, epoch_files)
        predictions_batch.append(predictions.tolist())

    # predictions_batch is [array([0])] (does not serialize)
    # predictions_batch is [[0]] (works)
    
    logging.warning('predictions_batch is %s', predictions_batch)
    predictions_batch_dict = {'predictions': predictions_batch}

    if not run_from_queue: 
        logging.warning('Return from normal execution')
        response = {
           "statusCode": 200,
           "body": json.dumps(predictions_batch_dict)
        }
    else:
        logging.warning('Return from queue execution')
        #logging.warning('predictions_batch decoded is %s', predictions_batch.decode('utf-8'))
        response = {
           "statusCode": 200,
           #"body": json.dumps(predictions_batch_dict, default=lambda x: x.decode('utf-8'))
           "body": json.dumps(predictions_batch_dict)
        }
        
    # response is {'statusCode': 200, 'body': '[[0]]'} fails from inferqueue
    # response is {'statusCode': 200, 'body': '{"predictions": [[0], [0]]}'} works from infer
    # response is {'statusCode': 200, 'body': '{"predictions": [[0]]}'} fails from inferqueue 

    
    logging.warning('response is %s', response)

    return response
