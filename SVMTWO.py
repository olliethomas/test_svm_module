import sys
import pandas as pd
from collections import namedtuple

from sklearn.svm import SVC

from cvxopt import matrix
DataTuple = namedtuple("DataTuple", ['x', 's', 'y'])


def run(train, test):
    A = matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2,3))
    clf = SVC(gamma='auto', random_state=888)
    clf.fit(train.x, train.y.values.ravel())
    return pd.DataFrame(clf.predict(test.x), columns=["preds"])


def load_dataframe(file_path):
    """Load a dataframe from a parquet file"""
    with open(file_path, 'rb') as file_obj:
        df = pd.read_feather(file_obj)
    return df


def load_data():
    """Load the data from the files"""
    train = DataTuple(
        x=load_dataframe((sys.argv[1])),
        s=load_dataframe((sys.argv[2])),
        y=load_dataframe((sys.argv[3])),
    )
    test = DataTuple(
        x=load_dataframe((sys.argv[4])),
        s=load_dataframe((sys.argv[5])),
        y=load_dataframe((sys.argv[6])),
    )
    return train, test


def save_predictions(predictions):
    """Save the data to the file that was specified in the commandline arguments"""
    if not isinstance(predictions, pd.DataFrame):
        df = pd.DataFrame(predictions, columns=["pred"])
    else:
        df = predictions
    pred_path = (sys.argv[7])
    df.to_feather(pred_path)


def main():
    """main method to run model"""
    train, test = load_data()
    save_predictions(run(train, test))


if __name__ == "__main__":
    main()
