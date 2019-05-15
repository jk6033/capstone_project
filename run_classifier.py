import classifier_utils as utils
import multiprocessing as mp


if __name__ == "__main__":
    # train classifier
    # fold_by = 4
    # clf = utils.train_randomforest(2)

    # # test classifier
    # utils.test_randomforest(clf=clf)

    b = utils.get_data_length(True)
    m = utils.get_data_length(False)
    len1 = next(b)
    len2 = next(m)
    print("binary train set: " + str(b))
    print("binary train set: " + str(m))