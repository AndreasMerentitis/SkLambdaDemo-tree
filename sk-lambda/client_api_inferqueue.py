import logging
import time

try:
    from urllib2 import HTTPError, urlopen
except ImportError:
    from urllib.request import urlopen
    from urllib.error import HTTPError

import json
import os
import subprocess

import click
import requests

@click.command(short_help='Classifies datapoints from json input file.')
def datapoint_classification_client():
    """Classifies datapoints from json input file.

    Takes one or more datapoints and returns the results from a TensorFlow"""
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    with open('data_input_small.json') as json_file:  
        points = json.load(json_file)

    logger.info('Predicting...')
    start_time = time.time()

    method = 'POST'
    headers = {'Content-Type': 'application/json'} 
    service = 'execute-api'
    url = 'https://z5ekmkl0t6.execute-api.eu-west-1.amazonaws.com/dev/inferqueue'
    region = 'eu-west-1'

    print('')
    print(url)
    print('')

    logger.info('Creating request...')
    
    auth = {}
    data = {}
    data['input'] = points['input']
    data['epoch'] = points['epoch']
    json_data = json.dumps(data)

    json_response = requests.request(method, url, auth=auth, data=json_data, headers=headers)

    end_time = time.time()
    
    # Extract text from JSON
    response = json.loads(json_response.text)
    flattened_response = [val for sublist in response for val in sublist]
    print(response)

    logger.info('End-to-end prediction (including network transfers) '
                'took {} seconds.'.format(
                    len(points), end_time - start_time))


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    datapoint_classification_client()




