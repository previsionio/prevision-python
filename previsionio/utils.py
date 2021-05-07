import operator
from .prevision_client import Client
from typing import Dict, List
import requests
import uuid
import json
import zipfile
import pandas as pd
import os
from collections import namedtuple
import numpy as np
from . import logger
import datetime
from math import ceil


def zip_to_pandas(pred_response: requests.Response) -> pd.DataFrame:
    temp_zip_path = '/tmp/ziptmp{}.zip'.format(str(uuid.uuid4()))

    with open(temp_zip_path, 'wb') as temp:
        for chunk in pred_response.iter_content(chunk_size=1024):
            if chunk:
                temp.write(chunk)

        temp.seek(0)

    with zipfile.ZipFile(temp_zip_path, 'r') as pred_zip:
        names = pred_zip.namelist()
        pred_zip.extractall('/tmp')
        pred_csv_path = '/tmp/' + names[0]
        data = pd.read_csv(pred_csv_path)
        os.remove(pred_csv_path)

    return data


class PrevisionException(Exception):
    pass


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.datetime64):
            return (datetime.datetime.fromtimestamp(obj.astype(datetime.datetime) // 10 ** 9)
                    .strftime('%Y-%m-%dT%H:%M:%s'))
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%dT%H:%M:%s')
        else:
            return super(NpEncoder, self).default(obj)


def parse_json(json_response) -> Dict:
    try:
        return json_response.json()
    except Exception as e:
        logger.error('[JSON PARSE] ' + json_response.text + '  --  ' + e.__repr__())
        raise


def get_pred_from_multiclassification(row, pred_prefix='pred_'):
    d = row.to_dict()
    preds_probas = {k: float(v) for k, v in d.items() if pred_prefix in k}
    pred = max(preds_probas.items(), key=operator.itemgetter(1))[0]
    pred = pred.replace(pred_prefix, '')
    return pred


EventTuple = namedtuple('EventTuple', 'name key value fail_checks')
EventTuple.__new__.__defaults__ = ((('state', 'failed'),),)


def is_null_value(value) -> bool:
    is_null = str(value) == 'null'
    is_nan = pd.isnull(value)
    return is_nan or is_null


def get_all_results(client: Client, endpoint: str, method) -> List[Dict]:
    resources = []
    batch: requests.Response = client.request(endpoint, method=method)
    json = parse_json(batch)
    print(json)
    meta = json['metaData']
    total_items = meta['totalItems']
    rows_per_page = meta['rowsPerPage']
    n_pages = ceil(total_items / rows_per_page)
    for n in range(1, n_pages + 1):
        batch = client.request(endpoint + "?page={}".format(n), method=method)
        resources.extend(parse_json(batch)['items'])
    return resources
