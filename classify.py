##
import json
import cPickle
import numpy as np


import tensorflow as tf
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
            PATH = "../result/multi/logs/train/result.json"
        else: 
            PATH = "../result/binary/logs/train/result.json"    
    else:
        if isMulti: 
            PATH = "../result/multi/logs/test/result.json"
        else: 
            PATH = "../result/binary/logs/test/result.json"

    print("Fetching label from " + PATH)
    
    with open(PATH) as f:
        json_file = json.load(f)

    label = json_file["answer"]
    return label
    
def concatenate_representation(vector1, vector2):
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)
    assert vector1.shape == vector2.shape
    print("Concatenating Vectors: vector1 shape of " + str(vector1.shape) + ", vector2 shape of " + str(vector2.shape))
    
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
    trainY = get_label(isMulti=True, isTrain=True)
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
    testY = get_label(isMulti=True, isTrain=False)
    assert len(testX) == len(testY)
    
    # calculate accuracy
    answer = clf.predict(testX)
    assert len(answer) == len(testY)
    correct = 0; total = 0
    for i in range(len(answer)):
        if answer[i] == np.asarray(testY)[i]: correct += 1
        total += 1
    accuracy = round(correct / total, 4)

    # store result
    test_jsonify = {
        "answer": testY,
        "output": answer.tolist(),
        "accuracy": accuracy,
        "entity": testX.tolist()
    }

    json.dump(test_jsonify, open('./bidir_dag_lstm_result/random_forest/result_raw.json', 'w'))

    print("Acc: " + str(accuracy))
    print("test complete!")


class Dataset:

    def __init__(self,data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass


    @property
    def data(self):
        return self._data

    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


def doMLP():
    trainX = np.asarray(get_representation(True))
    trainy_ = np.asarray(get_label(isMulti=True, isTrain=True))
    # one hot
    trainY = np.zeros((len(trainy_), 5))
    trainY[np.arange(len(trainy_)), trainy_] = 1

    testX = np.asarray(get_representation(False))
    testy_ = np.asarray(get_label(isMulti=True, isTrain=False))
    # one hot
    testY = np.zeros((len(testy_), 5))
    testY[np.arange(len(testy_)), testy_] = 1

    # Parameters
    learning_rate = 0.0001
    lambda_l2 = 0.001
    training_epochs = 1000
    batch_size = 10
    display_step = 10

    # Network Parameters
    n_hidden_1 = 4096 # 1st layer number of neurons
    n_hidden_2 = 4096 # 2nd layer number of neurons
    n_input = 1800 # 400 # data input (200 * 2)
    n_classes = 5 # total classes (0-4)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Hidden fully connected layer with 1024 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 1024 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    logits = tf.matmul(layer_2, weights['out']) + biases['out']

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    print("Accuracy: " + str(accuracy))
    
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    
    clipper = 5.0 # used to be 50
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tvars = tf.trainable_variables()
    
    assert lambda_l2 > 0.0
    if lambda_l2>0.0:
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if (v.get_shape().ndims > 1)])
        loss = loss + lambda_l2 * l2_loss # nan
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        batchX = Dataset(trainX)
        batchY = Dataset(trainY)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(trainX)/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = batchX.next_batch(batch_size)
                batch_y = batchY.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss], 
                        feed_dict={X: batch_x, Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch " + str('%04d'%(epoch+1)) + (": cost={:.9f}".format(avg_cost)))
        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:" + str(accuracy.eval({X: testX, Y: testY})))


if __name__ == "__main__":
    # your main code here
    # clf = train_randomforest()
    # test_randomforest(clf=clf)

    doMLP()