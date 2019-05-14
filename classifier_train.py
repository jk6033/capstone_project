import classifier_utils as utils

if __name__ == "__main__":
    
    trainX = utils.vec_concatenate(
                next(utils.get_multiclass_representation(True)), 
                next(utils.get_binary_representation(True)))
    utils.train_randomforest(
                trainX, 
                next(utils.get_multiclass_representation(True)))
