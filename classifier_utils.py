##
import numpy as np
import cPickle
from sklearn.ensemble import RandomForestClassifier
import json


def get_binary_representation(istrain=True): # isTrain should be always true; we will only be using multi class labels when testing
    if istrain:
        PATH = "../result/binary/logs/train/result.json"
    else:
        PATH = "../result/binary/logs/test/result.json"
    print("Getting Entity from " + PATH)
    with open(PATH) as f:
        json_file = json.load(f)
    
    # return entity
    yield json_file["entity"]

def get_multiclass_representation(istrain=True):
    if istrain: 
        PATH = "../result/multi/logs/train/result.json"
    else: 
        PATH = "../result/multi/logs/test/result.json"
    print("Getting Entity from " + PATH)
    with open(PATH) as f:
        json_file = json.load(f)

    # return (entity, answer)
    yield json_file["entity"]
    print("Getting Answer from " + PATH)
    yield json_file["answer"]

def vec_concatenate (vector1, vector2):
    assert np.asarray(vector1).shape == np.asarray(vector2).shape
    print("Concatenating Vectors...")
    vector = []
    for i in range(len(vector1)):
        temp = np.concatenate(vector1[i], vector2[i]).tolist()
        vector.append(temp)
    return np.asarray(vector)

def train_randomforest(trainX, trainY):
    assert len(trainX) == len(trainY)
    print("Building RF Classifer...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=15)
    clf.fit(trainX, trainY)

    # save randomforest
    with open('my_randomforest.pkl', 'wb') as f:
        cPickle.dump(clf, f)

    print("train completed!")

def test_randomforest(testX, testY):
    assert len(testX) == len(testY)
    # load randomforest
    with open('my_randomforest.pkl', 'rb') as f:
        clf = cPickle.load(f)

    # calculate accuracy
    answer = clf.predict(testX)
    assert len(answer) == len(testY)
    correct = 0; total = 0
    for i in range(len(answer)):
        if answer[i] == testY[i]: correct += 1
        total += 1
    accuracy = round(correct / total, 4)

    print("Acc: " + str(accuracy))
    print("test completed!")
    