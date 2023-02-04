try:
  import unzip_requirements
except ImportError:
  pass

import json
import os
import tarfile

import boto3
import tensorflow as tf
import numpy as np

import census_data

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']


def _easy_input_function(data_dict, batch_size=64):
    """
    data_dict = {
        '<csv_col_1>': ['<first_pred_value>', '<second_pred_value>']
        '<csv_col_2>': ['<first_pred_value>', '<second_pred_value>']
        ...
    }
    """

    # Convert input data to numpy arrays
    for col in data_dict:
        col_ind = census_data._CSV_COLUMNS.index(col)
        dtype = type(census_data._CSV_COLUMN_DEFAULTS[col_ind][0])
        data_dict[col] = np.array(data_dict[col],
                                        dtype=dtype)

    try: 
        labels = data_dict.pop('income_bracket')
    except:
        pass

    ds = tf.data.Dataset.from_tensor_slices((data_dict, labels))
    ds = ds.batch(64)

    return ds
    
    
def _predict_point(predict_input_point, epoch_files):
    """
    Makes predictions for a signle data point
    """

    # Download model from S3 and extract
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,'model.tar.gz'),
            FILE_DIR+'model.tar.gz')

    tarfile.open(FILE_DIR+'model.tar.gz', 'r').extractall(FILE_DIR)

    # Create feature columns
    wide_cols, deep_cols = census_data.build_model_columns()

    # Load model
    classifier = tf.estimator.LinearClassifier(
                    feature_columns=wide_cols,
                    model_dir=FILE_DIR+'tmp/model_'+epoch_files+'/',
                    warm_start_from=FILE_DIR+'tmp/model_'+epoch_files+'/')

    # Setup prediction
    predict_iter = classifier.predict(
                        lambda:_easy_input_function(predict_input_point))

    # Iterate over prediction and convert to lists
    predictions = []
    for prediction in predict_iter:
        for key in prediction:
            prediction[key] = prediction[key].tolist()

        predictions.append(prediction)

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
            predictions_batch.append(predictions)
    elif run_from_queue: 
        # Call from lambda queue
        predict_input_point = predict_input[0]
        if isinstance(predict_input_point, list):
           predict_input_point = predict_input_point[0]
        logging.warning('predict_input_point is %s', predict_input_point)
        predictions = _predict_point(predict_input_point, epoch_files)
        logging.warning('predictions is %s', predictions)
        predictions_batch.append(predictions)
    else: 
        # Direct call with one datapoint
        predict_input_point = predict_input
        predictions = _predict_point(predict_input_point, epoch_files)
        predictions_batch.append(predictions)

    if not run_from_queue: 
        logging.warning('Return from normal execution')
        response = {
           "statusCode": 200,
           "body": json.dumps(predictions_batch,
                            default=lambda x: x.decode('utf-8'))
        }
    else:
        logging.warning('Return from queue execution')
        response = {
           "statusCode": 200,
           "body": json.dumps(predictions_batch,
                            default=lambda x: x.decode('utf-8'))
        }
        
    logging.warning('response is %s', response)

    return response
