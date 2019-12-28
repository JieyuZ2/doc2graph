import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def evaluate_node_classification(args, labeled_data, embedding, repeat_times=5):
    nodes = labeled_data[:, 0]
    labels = labeled_data[:, 1][:, None]
    data = np.concatenate((embedding[nodes], labels), axis=1)
    split = int(len(labels) / repeat_times)

    accs = []
    for i in range(repeat_times):
        # cross validation
        p1, p2 = i * split, (i + 1) * split
        test = data[p1:p2, :]
        train1, train2 = data[:p1, :], data[p2:, :]
        train = np.concatenate([train1, train2])

        clf = SVC(gamma='auto').fit(train[:, :-1], train[:, -1])
        # clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, num_labels)).fit(train[:, :-1], train[:, -1])
        test_label = clf.predict(test[:, :-1])
        true_label = test[:, -1]
        accs.append(100 * accuracy_score(true_label, test_label))

    return np.mean(accs), np.std(accs)


def evaluate_taxonomy(args, test, model, level_by_level):
    return model.evaluate_taxo(test, level_by_level)


def evaluate_link_prediction(args, labeled_data, embedding, repeat_times=5):
    left_nodes = labeled_data[:, 0]
    right_nodes = labeled_data[:, 1]
    labels = labeled_data[:, [2]]
    x = sigmoid(np.sum(embedding[left_nodes]*embedding[right_nodes], axis=1))
    data = np.concatenate((x.reshape(-1, 1), labels), axis=1)
    split = int(len(labels) / repeat_times)

    aucs = []
    for i in range(repeat_times):
        # cross validation
        p1, p2 = i * split, (i + 1) * split
        test = data[p1:p2, :]
        aucs.append(100 * roc_auc_score(test[:, 1], test[:, 0]))

    return np.mean(aucs), np.std(aucs)
