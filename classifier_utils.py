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

    len_binary = len(json_binary["answer"])
    len_multi = len(json_multi["answer"])
    assert len_binary == len_multi

    yield len_binary

def get_binary_representation(batch_size, isTrain=True):
    if isTrain:
        PATH = "../result/binary/logs/train/result.json"
    else:
        PATH = "../result/binary/logs/test/result.json"
    print("Getting Entity from " + PATH)

    with open(PATH) as f:
        json_file = json.load(f)

    length = len(json_file["entity"])
    cumsum = 0
    while cumsum < length:
        try:
            entity = json_file["entity"][cumsum:(cumsum+batch_size)]
            print("Index fetched from " + str(cumsum) + " to " + str(cumsum+batch_size))
            yield entity
        except IndexError:
            entity = json_file["entity"][cumsum:-1]
            print("Index fetched from " + str(cumsum) + " to " + str(length-1))
            print("Binary classification data fetch complete!")
            yield entity
        finally:
            cumsum += batch_size

def get_multiclass_representation(batch_size, isTrain=True):
    if isTrain: 
        PATH = "../result/multi/logs/train/result.json"
    else: 
        PATH = "../result/multi/logs/test/result.json"
    print("Getting Entity from " + PATH)

    with open(PATH) as f:
        json_file = json.load(f)

    length = len(json_file["entity"])
    cumsum = 0
    while cumsum < length:
        try:
            entity = json_file["entity"][cumsum:(cumsum+batch_size)]
            print("Index fetched from " + str(cumsum) + " to " + str(cumsum+batch_size))
            yield entity
        except IndexError:
            entity = json_file["entity"][cumsum:-1]
            print("Index fetched from " + str(cumsum) + " to " + str(length-1))
            print("Multi classification data fetch complete!")
            yield entity
        finally:
            cumsum += batch_size

def get_multiclass_label(batch_size, isTrain=True):
    if isTrain: 
        PATH = "../result/multi/logs/train/result.json"
    else: 
        PATH = "../result/multi/logs/test/result.json"
    print("Getting Answer from " + PATH)
    
    with open(PATH) as f:
        json_file = json.load(f)

    length = len(json_file["entity"])
    cumsum = 0
    while cumsum < length:
        try:
            entity = json_file["entity"][cumsum:(cumsum+batch_size)]
            print("Index fetched from " + str(cumsum) + " to " + str(cumsum+batch_size))
            yield entity
        except IndexError:
            entity = json_file["entity"][cumsum:-1]
            print("Index fetched from " + str(cumsum) + " to " + str(length-1))
            print("Binary classification data fetch complete!")
            yield entity
        finally:
            cumsum += batch_size

def concatenate_representation(vector1, vector2):
    assert np.asarray(vector1).shape == np.asarray(vector2).shape
    print("Concatenating Vectors...")
    vector = []
    for i in range(len(vector1)):
        temp = np.concatenate(vector1[i], vector2[i]).tolist()
        vector.append(temp)
    return np.asarray(vector)

def train_randomforest(fold_by):
    # initialize random forest
    print("Initializing forest")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=15, warm_start=True)

    # calculate batch size
    g = get_data_length(True)
    data_size = next(g)
    batch_size = data_size // (fold_by -1)

    print("Trainging forest...")
    counter = 0
    binary_vector = get_binary_representation(batch_size, True)
    multi_vector = get_multiclass_representation(batch_size, True)
    multi_label = get_multiclass_label(batch_size, True)
    while counter < fold_by:
        vector1 = next(multi_vector)
        vector2 = next(binary_vector)

        trainX = concatenate_representation(vector1, vector2)
        trainY = next(multi_label)
        
        assert len(trainX) == len(trainY)
        clf.fit(trainX, trainY)
        clf.n_estimators += 50

    with open('my_randomforest.pkl', 'wb') as f:
        cPickle.dump(clf, f)

    return clf

def test_randomforest(model_path=None, clf=None): # one of them should be given
    print("Testing forest...")
    # validate input
    if model_path != None:
        with open(model_path, 'rb') as f:
            clf = cPickle.load(f)
    else:
        assert clf != None
    g = get_data_length(isTrain=False)
    length = next(g)

    vector1 = get_binary_representation(length, True)
    vector2 = get_multiclass_representation(length, True)

    testX = concatenate_representation(vector1, vector2)
    testY = get_multiclass_label(length, True)
    assert len(testX) == len(testY)
    
    # calculate accuracy
    answer = clf.predict(testX)
    assert len(answer) == len(testY)
    correct = 0; total = 0
    for i in range(len(answer)):
        if answer[i] == testY[i]: correct += 1
        total += 1
    accuracy = round(correct / total, 4)

    print("Acc: " + str(accuracy))
    print("test complete!")
    