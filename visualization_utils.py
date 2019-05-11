import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics  import confusion_matrix

# method that loads nencessay data from json
# model: "bidir_dag_lstm" or "gs_lstm"
# dataset: "train" or "validate" or "test"
# analysis: "accuracy"or "entity"
def load_data (model, dataset, analysis):

    data_path = "../nary-grn/" + model + "/logs/" + dataset + "/result.json"

    with open(data_path) as f:
        json_file = json.load(f)
            
    if analysis == "accuracy":
        answer = json_file["answer"]
        output = json_file["output"]
        return (answer, output)

    elif analysis == "entity":
        entity = json_file["entity"]
        return entity

# create confusion matrix to analyze the model's performance
# in this case, 5
def get_confusion_matrix (answer, output, class_num=5):     
    assert len(answer) == len(output)
    labels = [i for i in range(class_num)]
    matrix = confusion_matrix(answer, output, labels=labels)
    return matrix

# utility functions for confusion matrix
# axis: "col" or "row"
def normalize_confusion_matrix (matrix, axis="col"):
    normalized = np.round(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], 2)
    return normalized

def confusion_matrix_accuracy (matrix):
    acc = round(np.trace(matrix)/np.sum(matrix), 4)
    return acc

def get_tsne (entity):
    X = np.asarray(entity)
    # print("entity has a shape of", X.shape)

    model = TSNE(learning_rate = 100)
    # print("calculating TSNE...")
    transformed = model.fit_transform(X)

    return transformed

if __name__ == "__main__":

    model = ["bidir_dag_lstm", "gs_lstm"]
    dataset = ["train", "validate", "test"]
    # analysis = ["accuracy", "entity"]
    for m in model:
        result_jsonify = {}
        for d in dataset:
            print("working on: " + m +", " + d)

            answer, output = load_data(m, d, "accuracy")
            confusion_matrix = get_confusion_matrix(answer, output)
            normalized_conf = normalize_confusion_matrix(confusion_matrix)
            accuracy = confusion_matrix_accuracy(confusion_matrix)
                    
            entity = load_data(m, d, "entity")
            tsne_values = get_tsne(entity)

        if d == "train":
            result_jsonify["train_confusion_matrix"] = confusion_matrix
            result_jsonify["train_confusion_matrix_normalized"] = normalized_conf
            result_jsonify["train_accuracy"] = accuracy
            result_jsonify["train_tsne"] = tsne_values

        elif d == "validate":
            result_jsonify["valid_confusion_matrix"] = confusion_matrix
            result_jsonify["valid_confusion_matrix_normalized"] = normalized_conf
            result_jsonify["valid_accuracy"] = accuracy
            result_jsonify["valid_tsne"] = tsne_values

        elif d == "test":
            result_jsonify["test_confusion_matrix"] = confusion_matrix
            result_jsonify["test_confusion_matrix_normalized"] = normalized_conf
            result_jsonify["test_accuracy"] = accuracy
            result_jsonify["test_tsne"] = tsne_values
    
    path = "./" + model + "_result/result.json"
    json.dump(result_jsonify, open(path, 'w'))
