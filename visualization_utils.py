import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics  import confusion_matrix

# method that loads nencessay data from json
# model: "bidir_dag_lstm" or "gs_lstm"
# dataset: "train" or "validate" or "test"
# analysis: "accuracy"or "entity"
def load_data (model, dataset):

    data_path = "../result/" + model + "/logs/" + dataset + "/result.json"

    with open(data_path) as f:
        json_file = json.load(f)
    
    answer = json_file["answer"]
    output = json_file["output"]
    entity = json_file["entity"]
    return (answer, output, entity)

def get_tsne (entity):
    X = np.asarray(entity)
    # print("entity has a shape of", X.shape)

    model = TSNE(learning_rate = 100)
    # print("calculating TSNE...")
    transformed = model.fit_transform(X)

    return transformed.tolist()

def analyze(model, dataset):
    print("working on " + model)

    answer, output, entity = load_data(model, dataset)
    # confusion_matrix = get_confusion_matrix(answer, output)
    # normalized_conf = normalize_confusion_matrix(confusion_matrix)
    accuracy = calculate_accuracy(output, answer)
    tsne_values = get_tsne(entity)

    result = {}

    result["answer"] = answer
    result["output"] = output
    # result["confusion_matrix"] = confusion_matrix
    # result["confusion_matrix_normalized"] = normalized_conf
    result["accuracy"] = accuracy
    result["tsne"] = tsne_values

    path = "./bidir_dag_lstm_result/" + model + "/" + dataset + "/result.json"
    json.dump(result, open(path, 'w'))


if __name__ == "__main__":

    model = ["binary", "multi"]
    dataset = ["train", "test"]
    
    for i in range(len(model)):
        m = model[i]
        analyze(m, dataset[1])

        # for i in range(len(model)):
        #     analyze(model[i])