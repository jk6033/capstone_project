##
import numpy as np
# import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import json


def get_binary_representation(): # isTrain should be always true; we will only be using multi class labels when testing
    PATH = "../result/binary/logs/train/result.json"
    print("Getting Data from " + PATH)
    with open(PATH) as f:
        json_file = json.load(f)
    entity = json_file["entity"]
    return entity

def get_multiclass_representation(istrain=True):
    if istrain: 
        PATH = "../result/multi/logs/train/result.json"
    else: 
        PATH = "../result/multi/logs/test/result.json"
    print("Getting Data from " + PATH)
    with open(PATH) as f:
        json_file = json.load(f)
    entity = json_file["entity"]
    answer = json_file["answer"]
    return (entity, answer)

def vec_concatenate (vector1, vector2):
    assert np.asarray(vector1).shape == np.asarray(vector2).shape
    vector = []
    for i in range(len(vector1)):
        temp = np.concatenate(vector1[i], vector2[i]).tolist()
        vector.append(temp)
    return np.asarray(vector)

def randomforest(trainX, trainY, testX, testY):
    assert len(trainX) == len(trainY)
    assert len(testX) == len(testY)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=15)
    clf.fit(trainX, trainY)

    # calculate accuracy
    answer = clf.predict(testX)
    assert len(answer) == len(testY)
    correct = 0; total = 0
    for i in range(len(answer)):
        if answer[i] == testY[i]: correct += 1
        total += 1
    accuracy = round(correct / total, 4)

    print("Acc: " + str(accuracy))


if __name__ == "__main__":
    vector1, trainY = get_multiclass_representation(True)
    vector2 = get_binary_representation()

    testX, testY = get_multiclass_representation(False)

    print("Concatenating Vectors...")
    trainX = vec_concatenate(vector1, vector2)
    print("Building RF Classifer...")
    randomforest(trainX, trainY, testX, testY)

    