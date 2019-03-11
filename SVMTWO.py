from typing import Tuple

import pandas as pd

from sklearn.svm import SVC

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.utils import DataTuple
import pygame


class SVMEXAMPLE(InAlgorithm):
    """Support Vector Machine"""
    def run(self, train, test, sub_process=False):
        if sub_process:
            return self.run_threaded(train, test)

        pygame.init()
        clf = SVC(gamma='auto', random_state=888)
        clf.fit(train.x, train.y.values.ravel())
        return pd.DataFrame(clf.predict(test.x), columns=["preds"])

    @property
    def name(self):
        return "SVM"


def main():
    """main method to run model"""
    model = SVMEXAMPLE()
    train, test = model.load_data()
    model.save_predictions(model.run(train, test))


if __name__ == "__main__":
    main()
