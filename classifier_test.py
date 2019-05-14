import classifier_utils as utils


if __name__ == "__main__":
    binary = utils.get_binary_representation(False)
    multi = utils.get_multiclass_representation(False)
    
    testX = utils.vec_concatenate( next(multi), next(binary))
    utils.train_randomforest( testX, next(multi))