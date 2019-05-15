##
import json
import cPickle
import numpy as np
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier

def get_data_length(isTrain=True):
    if isTrain:
        PATH = "../result/binary/logs/train/result.json"
    else:
        PATH = "../result/binary/logs/test/result.json"
    print("Getting Entity from " + PATH)

    with open(PATH) as f:
        json_file = json.load(f)

    answer = len(json_file["answer"])

    yield answer

def get_binary_representation(range_start, range_end, isTrain=True):
    if isTrain:
        PATH = "../result/binary/logs/train/result.json"
    else:
        PATH = "../result/binary/logs/test/result.json"
    print("Getting Entity from " + PATH)

    with open(PATH) as f:
        json_file = json.load(f)

    try:
        entity = json_file["entity"][range_start:range_end]
        return entity
    except IndexError:
        entity = json_file["entity"][range_start:-1]
        print("Binary classification data fetch complete!")
        return entity

def get_multiclass_representation(range_start, range_end, isTrain=True):
    if isTrain: 
        PATH = "../result/multi/logs/train/result.json"
    else: 
        PATH = "../result/multi/logs/test/result.json"
    print("Getting Entity from " + PATH)

    with open(PATH) as f:
        json_file = json.load(f)

    try:
        entity = json_file["entity"][range_start:range_end]
        return entity
    except IndexError:
        entity = json_file["entity"][range_start:-1]
        print("Multi classification data fetch complete!")
        return entity

def get_multiclass_label(range_start, range_end, isTrain=True):
    if isTrain: 
        PATH = "../result/multi/logs/train/result.json"
    else: 
        PATH = "../result/multi/logs/test/result.json"
    print("Getting Answer from " + PATH)
    
    with open(PATH) as f:
        json_file = json.load(f)

    try:
        entity = json_file["entity"][range_start:range_end]
        return entity
    except IndexError:
        entity = json_file["entity"][range_start:-1]
        print("Multi classification label fetch complete!")
        return entity

def concatenate_representation(vector1, vector2):
    assert np.asarray(vector1).shape == np.asarray(vector2).shape
    print("Concatenating Vectors...")
    vector = []
    for i in range(len(vector1)):
        temp = np.concatenate(vector1[i], vector2[i]).tolist()
        vector.append(temp)
    return np.asarray(vector)

def train_randomforest(clf, range_start, range_end):
    print("Trainging forest...")
    vector1 = get_binary_representation(range_start, range_end, True)
    vector2 = get_multiclass_representation(range_start, range_end, True)

    trainX = concatenate_representation(vector1, vector2)
    trainY = get_multiclass_label(range_start, range_end)
    
    assert len(trainX) == len(trainY)
    clf.fit(trainX, trainY)

def train_multiprocess(fold_by):
    print("Initializing forest")
    # initialize random forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=15)

    # calculate batch size
    g = get_data_length(True)
    data_size = next(g)
    batch_size = data_size // (fold_by -1)

    arguements = [(clf, i*batch_size, (i+1)*batch_size)  for i in range(fold_by)]
    pool = mp.Pool(4)
    pool.map(train_randomforest, arguements) #arguement

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

    vector1 = get_binary_representation(0, -1, True)
    vector2 = get_multiclass_representation(0, -1, True)

    testX = concatenate_representation(vector1, vector2)
    testY = get_multiclass_label(0, -1)
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
    