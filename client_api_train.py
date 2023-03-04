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

    logger.info('Training...')
    start_time = time.time()

    method = 'POST'
    headers = {'Content-Type': 'application/json'} 
    service = 'execute-api'
    url1 = 'https://89dxba7gf1.execute-api.us-east-1.amazonaws.com/dev/upload'
    url2 = 'https://89dxba7gf1.execute-api.us-east-1.amazonaws.com/dev/train'
    region = 'us-east-1'

    print('')
    print(url1)
    print(url2)
    print('')

    logger.info('Creating request...')
    
    auth = {}

    json_response1 = requests.request(method, url1, auth=auth, headers=headers)
    json_response2 = requests.request(method, url2, auth=auth, headers=headers)

    end_time = time.time()
    
    # Extract text from JSON
    response1 = json.loads(json_response1.text)
    flattened_response1 = [val for sublist in response1 for val in sublist]
    print(response1)
    
    response2 = json.loads(json_response2.text)
    flattened_response2 = [val for sublist in response2 for val in sublist]
    print(response2)

    logger.info('End-to-end training (including network transfers) '
                'took {} seconds.'.format(end_time - start_time))


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    datapoint_classification_client()




