__author__ = 'jeremy'

import logging
import msgpack
import requests
import pandas as pd
import json

CLASSIFIER_ADDRESS='1.2.3.4'

logging.basicConfig(level=logging.INFO)

def bb_output_using_gunicorn(url_or_np_array):
    print('starting get_multilabel_output_using_nfc')
    multilabel_dict = nfc.pd(url_or_np_array, get_multilabel_results=True)
    logging.debug('get_multi_output:dict from falcon dict:'+str(multilabel_dict))
    if not multilabel_dict['success']:
        logging.warning('did not get nfc pd result succesfully')
        return
    multilabel_output = multilabel_dict['multilabel_output']
    logging.debug('multilabel output:'+str(multilabel_output))
    return multilabel_output #

def bb_output_yolo_using_api(url_or_np_array,CLASSIFIER_ADDRESS,roi=None,get_or_post='GET',query='file'):
    print('starting bb_output_api at addr '+str(CLASSIFIER_ADDRESS))
#    CLASSIFIER_ADDRESS =   # "http://13.82.136.127:8082/hls"
    print('using yolo api addr '+str(CLASSIFIER_ADDRESS))
    if isinstance(url_or_np_array,basestring): #got a url (use query= 'imageUrl') or filename, use query='file' )
        data = {query: url_or_np_array}
        print('using imageUrl as data')
    else:
        img_arr = url_or_np_array
        jsonified = pd.Series(img_arr).to_json(orient='values')

        data = {"image": jsonified} #this was hitting 'cant serialize' error
        print('using image as data')
    if roi:
        print("Make sure roi is a list in this order [x1, y1, x2, y2]")
        data["roi"] = roi
#    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    if get_or_post=='GET':
        result = requests.get(CLASSIFIER_ADDRESS,params=data)
    else:
        serialized_data = msgpack.dumps(data)
  #      result = requests.post(CLASSIFIER_ADDRESS,data=serialized_data)
        result = requests.post(CLASSIFIER_ADDRESS,data=data)

    if result.status_code is not 200:
       print("Code is not 200")
#     else:
#         for chunk in result.iter_content():
#             print(chunk)
# #            joke = requests.get(JOKE_URL).json()["value"]["joke"]

#    resp = requests.post(CLASSIFIER_ADDRESS, data=data)
    c = result.content
    #content should be roughly in form
#    {"data":
    # [{"confidence": 0.366, "object": "car", "bbox": [394, 49, 486, 82]},
    # {"confidence": 0.2606, "object": "car", "bbox": [0, 116, 571, 462]}, ... ]}
    if not 'data' in c:
        print('didnt get data in result from {} on sendng {}'.format(CLASSIFIER_ADDRESS,data))
    return c

def detect(img_arr, CLASSIFIER_ADDRESS,roi=[]):
    print('using addr '+str(CLASSIFIER_ADDRESS))
    data = {"image": img_arr}
    if roi:
        print "Make sure roi is a list in this order [x1, y1, x2, y2]"
        data["roi"] = roi
    serializer = json
    serialized_data = serializer.dumps(data)
#    serialized_data = msgpack.dumps(data)
#    resp = requests.post(YOLO_HLS_ADDRESS, data=data)
    resp = requests.post(CLASSIFIER_ADDRESS, data=serialized_data)
    print('resp from hls:'+str(resp))
    print('respcont from hls:'+str(resp.content))
    print('respctest from hls:'+str(resp.text))
    return msgpack.loads(resp.content)

