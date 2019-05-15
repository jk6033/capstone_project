##
import json
import cPickle
import numpy as np
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier

def get_data_length(isTrain=True):
    if isTrain:
        PATH_binary = "../result/binary/logs/train/result.json"
        PATH_multi = "../result/multi/logs/train/result.json"
    else:
        PATH_binary = "../result/binary/logs/test/result.json"
        PATH_multi = "../result/multi/logs/test/result.json"
    print("Getting data length from " + PATH_binary)
    with open(PATH_binary) as b:
        json_binary = json.load(b)
    print("Getting data length from " + PATH_multi)
    with open(PATH_multi) as m:
        json_multi = json.load(m)

    len_binary = len(json_binary["entity"])
    print("Binary Data Length: " + str(len_binary))
    len_multi = len(json_multi["entity"])
    print("Multi Class Data Length: " + str(len_multi))
    assert len_binary == len_multi

    return len_binary

def get_representation(isTrain=True):
    if isTrain:
        PATH_binary = "../result/binary/logs/train/result.json"
        PATH_multi = "../result/multi/logs/train/result.json"
    else:
        PATH_binary = "../result/binary/logs/test/result.json"
        PATH_multi = "../result/multi/logs/test/result.json"
    print("Fetching entity from " + PATH_binary)
    with open(PATH_binary) as b:
        json_binary = json.load(b)
    print("Fetching entity from " + PATH_multi)
    with open(PATH_multi) as m:
        json_multi = json.load(m)

    vector1 = json_binary["entity"]
    vector2 = json_multi["entity"]
    
    representation = concatenate_representation(vector1, vector2)

    return representation

def get_label(isMulti=True, isTrain=True):
    if isTrain:
        if isMulti: 
            PATH = "../result/binary/logs/train/result.json"
        else: 
            PATH = "../result/multi/logs/train/result.json"    
    else:
        if isMulti: 
            PATH = "../result/binary/logs/test/result.json"
        else: 
            PATH = "../result/multi/logs/test/result.json"

    print("Fetching label from " + PATH)
    
    with open(PATH) as f:
        json_file = json.load(f)

    label = json_file["answer"]
    return label
    
def concatenate_representation(vector1, vector2):
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)
    assert vector1.shape == vector2.shape
    print("Concatenating Vectors...")
    
    representation = np.empty((vector1.shape[0], vector1.shape[1]*2))
    for i in range(len(vector1)):
        representation[i] = np.concatenate((vector1[i], vector2[i]))
        
    return representation

def train_randomforest():
    # initialize random forest
    print("Initializing forest")
    clf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=30)

    print("Trainging forest...")
    trainX = get_representation(True)
    trainY = get_label(True, True)
    assert len(trainX) == len(trainY)

    clf.fit(trainX, trainY)

    with open('./bidir_dag_lstm_result/random_forest/RF_model.pkl', 'wb') as f:
        cPickle.dump(clf, f)

    return clf

def test_randomforest(model_path=None, clf=None): # either of them should be given
    print("Testing forest...")
    # validate input
    if model_path != None:
        with open(model_path, 'rb') as f:
            clf = cPickle.load(f)
    else:
        assert clf != None

    testX = get_representation(False)
    testY = get_label(False)
    assert len(testX) == len(testY)
    
    # calculate accuracy
    answer = clf.predict(testX)
    assert len(answer) == len(testY)
    correct = 0; total = 0
    for i in range(len(answer)):
        if answer[i] == testY[i]: correct += 1
        total += 1
    accuracy = round(correct / total, 4)

    # store result
    test_jsonify = {
        "answer": testY,
        "output": answer,
        "accuracy": accuracy,
        "entity": testX
    }

    json.dump(test_jsonify, open('./bidir_dag_lstm_result/random_forest/result.json', 'w'))

    print("Acc: " + str(accuracy))
    print("test complete!")

if __name__ == "__main__":
    # your main code here
    clf = train_randomforest()
    test_randomforest(clf=clf)