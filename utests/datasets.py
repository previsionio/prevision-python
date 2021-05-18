import os
import argparse
import datetime
import glob
import numpy as np
import pandas as pd
import requests

URLS = {
    "titanic": [
        "https://raw.githubusercontent.com/ashishpatel26/Titanic-Machine-Learning-from-Disaster/master/input/train.csv",
        "https://raw.githubusercontent.com/ashishpatel26/Titanic-Machine-Learning-from-Disaster/master/input/test.csv"],
    "forest_fire": ["https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/"],
    "housing": ["https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"],
    "iris": ["https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/" +
             "639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv"]
}


def fetch_dataset(url, target=None, name=None, stream=False):
    """
    Scrape titanic dataset from the given url
    :param url: url to row dataset (REQUIRED)
    :param target: if given it indicates the local path
                    to save the retrieved dataset
                    (OPTIONAL)
    :param name: if given the retrieved dataset will be saved with the given name
    :param stream: write in stream mode (recommended for very large datasets)
    :return: -pandas dataframe of the retrieved dataset
             -None if Request error or dataset not found
    """
    if not name:
        name = url.split('/')[-1].split('.')[0]

    csv_file = os.path.join('/tmp', name + ".csv")
    if target:
        if not os.path.isdir(target):
            os.makedirs(target)
        csv_file = os.path.join(target, name + ".csv")

    if not stream:
        r = requests.get(url)
        if r.ok:
            with open(csv_file, 'wb') as f:
                f.write(r.content)
            result = pd.read_csv(csv_file)
            os.remove(csv_file)
            return result
        return None

    r = requests.get(url, stream=True)
    if r.ok:
        # read chunk by chunk
        handle = open(csv_file, "wb")
        for chunk in r.iter_content(chunk_size=512):
            if chunk:
                handle.write(chunk)
        handle.close()
        result = pd.read_csv(csv_file)
        os.remove(csv_file)
        return result
    return None


def make_supervised_datasets(path,
                             n_smp=100, n_feat_reg=2, n_feat_classif=2,
                             n_feat_multiclassif=2, n_feat_ts=1):
    if not os.path.exists(path):
        os.mkdir(path)

    # regression
    X_reg = np.random.rand(n_smp, n_feat_reg)
    y_reg = np.random.rand(n_smp)
    reg = pd.DataFrame(X_reg,
                       columns=['feat_{}'.format(i) for i in range(n_feat_reg)])
    reg['target'] = y_reg
    reg_path = os.path.join(path, 'regression.csv')
    reg.to_csv(reg_path, index=False)

    # classification
    X_classif = np.random.rand(n_smp, n_feat_classif)
    y_classif = np.random.randint(0, 2, size=n_smp)
    classif = pd.DataFrame(X_classif,
                           columns=['feat_{}'.format(i) for i in range(n_feat_classif)])
    classif['target'] = y_classif
    classif_path = os.path.join(path, 'classification.csv')
    classif.to_csv(classif_path, index=False)

    # multiclassification
    X_multiclassif = np.random.rand(n_smp, n_feat_multiclassif)
    y_multiclassif = np.random.randint(0, 3, size=n_smp)
    multiclassif = pd.DataFrame(X_multiclassif,
                                columns=['feat_{}'.format(i) for i in
                                         range(n_feat_multiclassif)])
    multiclassif['target'] = y_multiclassif
    mclassif_path = os.path.join(path, 'multiclassification.csv')
    multiclassif.to_csv(mclassif_path, index=False)

    # time series
    X_ts = np.random.uniform(-5, 5, size=(n_smp, n_feat_ts))
    y_ts = np.random.uniform(-5, 5, size=(n_smp,))
    ts = pd.DataFrame(X_ts, columns=['feat_{}'.format(i) for i in range(n_feat_ts)])
    ts['target'] = y_ts
    ts['time'] = pd.date_range(datetime.datetime(2018, 1, 1), periods=len(X_ts), freq='1D')
    ts_path = os.path.join(path, 'ts.csv')
    ts.to_csv(ts_path, index=False)

    # big
    # n_smp_big = 10000
    # n_feat_big = 100
    # X_big = np.random.rand(n_smp_big, n_feat_big)
    # y_big = np.random.randint(0, 2, size=n_smp_big)
    # big = pd.DataFrame(X_big, columns=['feat_{}'.format(i) for i in range(n_feat_big)])
    # big['target'] = y_big
    # big_path = os.path.join(path, 'big.csv')
    # big.to_csv(big_path, index=False)

    return {
        'regression': reg_path,
        'classification': classif_path,
        'multiclassification': mclassif_path,
        # 'timeseries': ts_path,
        # 'big': big_path
    }


def remove_datasets(path):
    if os.path.exists(path):
        for f in glob.glob(os.path.join(path, '*.csv')):
            os.remove(f)
        os.rmdir(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_folder', default='data',
                        help='folder in which datasets will be saved')
    parser.add_argument('-n_samples', type=int, default='100',
                        help='size of datasets created')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_folder = args.data_folder
    n_samples = args.n_samples
    remove_datasets(data_folder)
    make_supervised_datasets(data_folder, n_smp=n_samples)


if __name__ == '__main__':
    main()
