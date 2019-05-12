import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics  import confusion_matrix

# method that loads nencessay data from json
# model: "bidir_dag_lstm" or "gs_lstm"
# dataset: "train" or "validate" or "test"
# analysis: "accuracy"or "entity"
def load_data (model, dataset):

    data_path = "../nary-grn/" + model + "/logs/" + dataset + "/result.json"

    with open(data_path) as f:
        json_file = json.load(f)
    
    answer = json_file["answer"]
    output = json_file["output"]
    entity = json_file["entity"]
    return (answer, output, entity)

# create confusion matrix to analyze the model's performance
# in this case, 5
def get_confusion_matrix (answer, output, class_num=5):     
    assert len(answer) == len(output)
    labels = np.asarray([i for i in range(class_num)])
    matrix = confusion_matrix(np.asarray(answer), np.asarray(output), labels=labels)
    return matrix.tolist()

# utility functions for confusion matrix
# axis: "col" or "row"
def normalize_confusion_matrix (matrix, axis="col"):
    matrix = np.asarray(matrix)
    normalized = np.round(matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis], 2)
    return normalized.tolist()

def confusion_matrix_accuracy (matrix):
    matrix = np.asarray(matrix)
    acc = round(np.trace(matrix)/np.sum(matrix), 4)
    return [acc]

def get_tsne (entity):
    X = np.asarray(entity)
    # print("entity has a shape of", X.shape)

    model = TSNE(learning_rate = 100)
    # print("calculating TSNE...")
    transformed = model.fit_transform(X)

    return transformed.tolist()

def analyze(model, dataset):
    print("working on " + model +" & " + dataset)

    answer, output, entity = load_data(model, dataset)
    confusion_matrix = get_confusion_matrix(answer, output)
    normalized_conf = normalize_confusion_matrix(confusion_matrix)
    accuracy = confusion_matrix_accuracy(confusion_matrix)
    tsne_values = get_tsne(entity)

    result = {}

    result["answer"] = answer
    result["output"] = output
    result["confusion_matrix"] = confusion_matrix
    result["confusion_matrix_normalized"] = normalized_conf
    result["accuracy"] = accuracy
    result["tsne"] = tsne_values

    path = "./" + model + "_result/" + dataset + "/result.json"
    json.dump(result, open(path, 'w'))


if __name__ == "__main__":

    model = ["bidir_dag_lstm", "gs_lstm"]
    dataset = ["validate", "test", "train"]
    

    # for i in range(len(model)):
    #     m = model[i]
    #     for j in range(len(dataset)):
    #         d = dataset[j]

    #         analyze(m, d)
    for i in range(len(dataset)):
        analyze(model[1], dataset[i])