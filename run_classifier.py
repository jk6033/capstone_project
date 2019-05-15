import classifier_utils as utils
import multiprocessing as mp


if __name__ == "__main__":
    # train classifier
    fold_by = 4
    clf = utils.train_multiprocess(fold_by)

    # test classifier
    utils.test_randomforest(clf)