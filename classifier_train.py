import classifier_utils as utils

if __name__ == "__main__":
    binary = utils.get_binary_representation(True)
    multi = utils.get_multiclass_representation(True)
    
    trainX = utils.vec_concatenate( next(multi), next(binary))
    utils.train_randomforest( trainX, next(multi))
