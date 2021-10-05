import os
import json
import sys

__CRNN_CONFIG__ = None

def get_config(file:str=None, mode:str=None) -> dict:
    global __CRNN_CONFIG__
    if file is None:
        file = os.path.dirname(os.path.realpath(__file__)) + '/config.json'
    else:
        file+='/config.json'
    if __CRNN_CONFIG__ is None:
        try:
            with open(file, 'r') as fid:
                __CRNN_CONFIG__ = json.load(fid)
        except:
            print('Unexpected Error:', sys.exc_info())
    __CRNN_CONFIG__['WORKMODE'] = mode
    return __CRNN_CONFIG__