import classifier_utils as utils


if __name__ == "__main__":
    vector1, testY = utils.get_multiclass_representation(False)
    vector2 = utils.get_binary_representation(False)
    testX = utils.vec_concatenate(vector1, vector2)
    utils.test_randomforest(testX, testY)