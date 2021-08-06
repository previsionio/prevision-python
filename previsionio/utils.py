from enum import Enum
import operator
from typing import Dict, List, Tuple, Union
import requests
import uuid
import json
import zipfile
import pandas as pd
import os
import numpy as np
from requests.models import Response
from . import logger, config
import datetime
from math import ceil


def zip_to_pandas(pred_response: requests.Response, separator=None) -> pd.DataFrame:
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
        data = pd.read_csv(pred_csv_path, sep=separator, engine='python')
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


def parse_json(json_response: Response) -> Dict:
    try:
        return json_response.json()
    except Exception as e:
        logger.error('[JSON PARSE] ' + json_response.text + '  --  ' + e.__repr__())
        raise


def to_json(obj):
    if isinstance(obj, Enum):
        return to_json(obj.value)
    elif isinstance(obj, list):
        obj_list = []
        for e in obj:
            obj_list.append(to_json(e))
        return obj_list
    elif isinstance(obj, dict):
        obj_d = {}
        for key, value in obj.items():
            obj_d[key] = to_json(value)
        return obj_d
    elif hasattr(obj, '__dict__'):
        obj_dict = {}
        for key, value in obj.__dict__.items():
            if value:
                if hasattr(obj, 'config') and key in obj.config:
                    key = obj.config[key]
                obj_dict[key] = to_json(value)
        return obj_dict
    return obj


def get_pred_from_multiclassification(row, pred_prefix: str = 'pred_'):
    d = row.to_dict()
    preds_probas = {k: float(v) for k, v in d.items() if pred_prefix in k}
    pred = max(preds_probas.items(), key=operator.itemgetter(1))[0]
    pred = pred.replace(pred_prefix, '')
    return pred


class EventTuple():
    def __init__(self, name: str,
                 success_checks: Union[Tuple[str, str], List[Tuple[str, str]]] = ('state', 'done'),
                 fail_checks: Union[Tuple[str, str], List[Tuple[str, str]]] = ('state', 'failed')):
        assert name is not None
        self.name = name

        assert success_checks is not None
        self.success_checks = success_checks
        if isinstance(success_checks, Tuple):
            self.success_checks = [success_checks]
        assert len(set([key for key, _ in self.success_checks])) == len(
            self.success_checks), "requiring same key for success check"

        assert fail_checks is not None
        self.fail_checks = fail_checks
        if isinstance(fail_checks, Tuple):
            self.fail_checks = [fail_checks]

    def is_failure(self, json: Dict):
        """
            if any fail condition pass, it's a failure
        """
        for key, val in self.fail_checks:
            if json.get(key) == val:
                return True
        return False

    def is_success(self, json: Dict):
        """
            all success condition need to be there to pass
        """
        result = True
        for key, val in self.success_checks:
            if json.get(key) != val:
                return False
        return result


def is_null_value(value) -> bool:
    is_null = str(value) == 'null'
    is_nan = pd.isnull(value)
    return is_nan or is_null


def get_all_results(client, endpoint: str, method) -> List[Dict]:
    resources = []
    batch: requests.Response = client.request(endpoint, method=method)
    json = parse_json(batch)
    meta = json['metaData']
    total_items = meta['totalItems']
    rows_per_page = meta['rowsPerPage']
    n_pages = ceil(total_items / rows_per_page)
    for n in range(1, n_pages + 1):
        url = endpoint + "?page={}".format(n)
        batch = client.request(url, method=method)
        resources.extend(parse_json(batch)['items'])
    return resources


def handle_error_response(
    resp: Response,
    url: str,
    data: Union[Dict, List] = None,
    files: Dict = None,
    message_prefix: str = None,
    n_tries: int = 1,
    additional_log: str = None,
):
    if resp.status_code not in config.success_codes:
        message = "Error {}: '{}' reaching url: '{}'".format(
            resp.status_code, resp.text, url)
        if n_tries > 1:
            message += " after {} tries".format(n_tries)
        if data:
            message += " with data: {}".format(data)
        if files:
            message += " with files: {}".format(files)
        if message_prefix:
            message = message_prefix + ' failure\n' + message
        logger.error(message)
        if additional_log:
            logger.error(additional_log)
        raise PrevisionException(message)
