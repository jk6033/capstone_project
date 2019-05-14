import classifier_utils as utils


if __name__ == "__main__":
    vector1, trainY = utils.get_multiclass_representation(True)
    vector2 = utils.get_binary_representation(True)
    trainX = utils.vec_concatenate(vector1, vector2)
    utils.train_randomforest(trainX, trainY)
