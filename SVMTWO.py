import sys
import pandas as pd
from collections import namedtuple

from sklearn.svm import SVC

import pygame

DataTuple = namedtuple("DataTuple", ['x', 's', 'y'])


def run(self, train, test, sub_process=False):
    if sub_process:
        return self.run_threaded(train, test)

    pygame.init()
    clf = SVC(gamma='auto', random_state=888)
    clf.fit(train.x, train.y.values.ravel())
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def load_dataframe(file_path):
    """Load a dataframe from a parquet file"""
    with file_path.open('rb') as file_obj:
        df = pd.read_parquet(file_obj)
    return df


def load_data():
    """Load the data from the files"""
    train = DataTuple(
        x=load_dataframe((sys.argv[0])),
        s=load_dataframe((sys.argv[1])),
        y=load_dataframe((sys.argv[2])),
    )
    test = DataTuple(
        x=load_dataframe((sys.argv[3])),
        s=load_dataframe((sys.argv[4])),
        y=load_dataframe((sys.argv[5])),
    )
    return train, test


def save_predictions(predictions):
    """Save the data to the file that was specified in the commandline arguments"""
    if not isinstance(predictions, pd.DataFrame):
        df = pd.DataFrame(predictions, columns=["pred"])
    else:
        df = predictions
    pred_path = (sys.argv[6])
    df.to_parquet(pred_path, compression=None)


def main():
    """main method to run model"""
    train, test = load_data()
    save_predictions(run(train, test))


if __name__ == "__main__":
    main()
