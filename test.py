#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
from collections import Counter

import tqdm

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# seed value
# (ensures consistent dataset splitting between runs)
SEED = 0


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error("That directory {} does not exist!".format(x))
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7, 
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    """Load json feature files produced from feature extraction.

    The device label (MAC) is identified from the directory in which the feature file was found.
    Returns X and Y as separate multidimensional arrays.
    The instances in X contain only the first 6 features.
    The ports, domain, and cipher features are stored in separate arrays for easier process in stage 0.

    Parameters
    ----------
    root : str
           Path to the directory containing samples.
    min_samples : int
                  The number of samples each class must have at minimum (else it is pruned).
    max_samples : int
                  Stop loading samples for a class when this number is reached.

    Returns
    -------
    features_misc : numpy array
    features_ports : numpy array
    features_domains : numpy array
    features_ciphers : numpy array
    labels : numpy array
    """
    X = []
    X_p = []
    X_d = []
    X_c = []
    Y = []

    port_dict = dict()
    domain_set = set()
    cipher_set = set()

    # create paths and do instance count filtering
    fpaths = []
    fcounts = dict()
    for rt, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(rt, fname)
            label = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            if name.startswith("features") and name.endswith(".json"):
                fpaths.append((path, label, name))
                fcounts[label] = 1 + fcounts.get(label, 0)

    # load samples
    processed_counts = {label:0 for label in fcounts.keys()}
    for fpath in tqdm.tqdm(fpaths):
        path = fpath[0]
        label = fpath[1]
        if fcounts[label] < min_samples:
            continue
        if processed_counts[label] >= max_samples:
            continue
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"],
                        features["flow_duration"],
                        features["flow_rate"],
                        features["sleep_time"],
                        features["dns_interval"],
                        features["ntp_interval"]]
            X.append(instance)
            X_p.append(list(features["ports"]))
            X_d.append(list(features["domains"]))
            X_c.append(list(features["ciphers"]))
            Y.append(label)
            domain_set.update(list(features["domains"]))
            cipher_set.update(list(features["ciphers"]))
            for port in set(features["ports"]):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > 10:
            port_set.add(port)

    # map to wordbag
    print("Generating wordbags ... ")
    for i in tqdm.tqdm(range(len(Y))):
        X_p[i] = list(map(lambda x: X_p[i].count(x), port_set))
        X_d[i] = list(map(lambda x: X_d[i].count(x), domain_set))
        X_c[i] = list(map(lambda x: X_c[i].count(x), cipher_set))

    return np.array(X).astype(float), np.array(X_p), np.array(X_d), np.array(X_c), np.array(Y)


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier to analyze the 'bag of words' seen in the ports/domain/ciphers features.
    Returns the prediction results for the training and testing datasets as an array of tuples in which each row
      represents a data instance and each tuple is composed as the predicted class and the confidence of prediction.

    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    C_tr : numpy array
           Prediction results for training samples.
    C_ts : numpy array
           Prediction results for testing samples.
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)

    # produce class and confidence for training samples
    C_tr = classifier.predict_proba(X_tr)
    C_tr = [(np.argmax(instance), max(instance)) for instance in C_tr]

    # produce class and confidence for testing samples
    C_ts = classifier.predict_proba(X_ts)
    C_ts = [(np.argmax(instance), max(instance)) for instance in C_ts]

    return C_tr, C_ts


def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 of the classification procedure:
        process each multinomial feature using naive bayes
        return the class prediction and confidence score for each instance feature

    Parameters
    ----------
    Xp_tr : numpy array
           Array containing training (port) samples.
    Xp_ts : numpy array
           Array containing testing (port) samples.
    Xd_tr : numpy array
           Array containing training (port) samples.
    Xd_ts : numpy array
           Array containing testing (port) samples.
    Xc_tr : numpy array
           Array containing training (port) samples.
    Xc_ts : numpy array
           Array containing testing (port) samples.
    Y_tr : numpy array
           Array containing training labels.
    Y_ts : numpy array
           Array containing testing labels

    Returns
    -------
    resp_tr : numpy array
              Prediction results for training (port) samples.
    resp_ts : numpy array
              Prediction results for testing (port) samples.
    resd_tr : numpy array
              Prediction results for training (domains) samples.
    resd_ts : numpy array
              Prediction results for testing (domains) samples.
    resc_tr : numpy array
              Prediction results for training (cipher suites) samples.
    resc_ts : numpy array
              Prediction results for testing (cipher suites) samples.
    """
    # perform multinomial classification on bag of ports
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)

    # perform multinomial classification on domain names
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)

    # perform multinomial classification on cipher suites
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts

class DecisionTree:
    """
    Splits the data into subgroups based on feature thresholds
    Intended to create increasingly pure groups or groups mostly containing the same class
    """
    def __init__(self, maxDepth = 20, minNode = 5, featureSubcount = None, rng = None):
         # Maximum depth the tree can grow to prevent overfiting
        self.maxDepth = maxDepth       
        # Smallest number of samples allowed in a node before stopping
        self.minNode = minNode
        # If set, only a random subset of features are considered per split          
        self.feature_subcount = featureSubcount  
        # Random number generator for reproducibility that defaults to a fixed seed
        self.rng = np.random.RandomState(0) if rng is None else rng 
        # Stores the built tree structure.
        self.root = None #

    def gini(self, y):
        """
        Computes the Gini impurity for a set of labels
        The lower the computation the more pure the node

        """ 
        if len(y) == 0:
            return 0
        
        # Counter the number of each class occurrence 
        counts = np.bincount(y)
        # Converts the count into probabilities
        probabilities = counts[counts > 0] / len(y)
        # Computes the Gini formula
        return 1 - np.sum(probabilities ** 2)

    def bestSplit(self, X, y):
        """
        Tries splitting on each feature and threshold
        Selects the split that gives the lowest weighted Gini impurity

        """
        numSamples, numFeatures = X.shape
        # Sets the parent Gini node to that Gini value prior to any splitting
        parentGini = self.gini(y)

        # Decide which features to attempt, either all feature or a random subset
        if self.feature_subcount is None:
            features = range(numFeatures)
        else:
            features = self.rng.choice(numFeatures, self.feature_subcount, replace = True)

        # Variables intended to store and track the best result
        bestFeature, bestThreshold = None, None
        bestLeftIndex, bestRightIndex = None, None
        # Starts with zero improvement
        bestGini = parentGini  

        # Attempts to conduct splitting on each feature
        for feat in features:
            # Sort the rows of samples by the feature values, whichmakes threshold search easier
            sortedIndex = np.argsort(X[:, feat])
            xSorted = X[sortedIndex, feat]

            # Thresholds occur where the feature value switches
            for i in range(1, numSamples):
                if xSorted[i] == xSorted[i - 1]:
                    # Skips if no new threshold is found
                    continue
                thr = (xSorted[i] + xSorted[i - 1]) / 2

                left_mask = (X[:, feat] <= thr)
                right_mask = ~left_mask
                yLeft, yRight = y[left_mask], y[right_mask]

                # Avoids any empty splits
                if len(yLeft) == 0 or len(yRight) == 0:
                    continue

                # The weighted gini of the split
                giniLeft = self.gini(yLeft)
                giniRight = self.gini(yRight)
                giniSplit = (len(yLeft) / numSamples) * giniLeft + (len(yRight) / numSamples) * giniRight

                # Keeps the best split when the lower gini is optimal
                if giniSplit < bestGini:
                    bestGini = giniSplit
                    bestFeature = feat
                    bestThreshold = thr
                    bestLeftIndex = np.where(left_mask)[0]
                    bestRightIndex = np.where(right_mask)[0]

        return bestFeature, bestThreshold, bestLeftIndex, bestRightIndex, bestGini, parentGini

    def build(self, X, y, depth):
        # Creates a node that defaults to predicting the majority label
        node = {
            "pred": np.argmax(np.bincount(y)),  # The majority class label
            "feat": None,
            "thr": None,
            "left": None,
            "right": None
        }

        # Stops if max depth is reached or there are too few samples to split further
        if depth >= self.maxDepth or len(y) < self.minNode:
            return node

        feat, thr, leftIndex, rightIndex, bestGini, parentGini = self.bestSplit(X, y)

        # If there is no helpful split, stop and return to the leaf node
        if feat is None or bestGini >= parentGini:
            return node

        # Otherwise, store the split and build the left and right subtrees
        node["feat"] = feat
        node["thr"] = thr
        # Uses recursion to build the left and right branches
        node["left"] = self.build(X[leftIndex], y[leftIndex], depth + 1)
        node["right"] = self.build(X[rightIndex], y[rightIndex], depth + 1)
        return node

    def fit(self, X, y):
        # Ensures that the labels are integers
        y = y.astype(int) 
        # Trains the decision tree
        self.root = self.build(X, y, depth = 0)
        return self

    def predictOne(self, x):
        # Predicts one example by using its splits to traverse the tree 
        node = self.root
        while node["feat"] is not None:  
            if x[node["feat"]] <= node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["pred"]

    def predict(self, X):
        # Predict labels for entire dataset
        return np.array([self.predictOne(sample) for sample in X])


class RandomForest:
    """
    Random Forest occurs with multiple decision trees andn then voting
    Each tree sees a random subset of training data (bootstrap sampling) and features (per-split feature subsampling)

    """
    def __init__(self, numTrees = 100, maxDepth = 20, minNode = 5, dataFraction = 0.7, featureSubcount = None):
        self.numTrees = numTrees
        self.maxDepth = maxDepth
        self.minNode = minNode
        self.dataFraction = dataFraction
        self.featureSubcount = featureSubcount
        self.trees = []
        self.rng = np.random.RandomState(0)

    def fit(self, X, y):
        n = len(y)
        bootstrap_size = max(1, int(self.dataFraction * n))

        for _ in range(self.numTrees):
            # Bootstrap sampling 
            idx = self.rng.choice(n, size = bootstrap_size, replace = True)
            X_sub, y_sub = X[idx], y[idx]

            tree = DecisionTree(
                maxDepth = self.maxDepth,
                minNode = self.minNode,
                featureSubcount = self.featureSubcount,
                rng = self.rng
            )
            tree.fit(X_sub, y_sub)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # Collect predictions from each tree
        treePredictions = []
        for tree in self.trees:
            # Retrieve the tree's predictions for all samples
            predictions = tree.predict(X)  
            treePredictions.append(predictions)
        treePredictions = np.array(treePredictions)

        # Selects the most common label per sample
        final = []
        for col in treePredictions.T:  # Iterates per sample
            final.append(np.argmax(np.bincount(col)))
        return np.array(final)
    
def do_stage_1(X_tr, X_ts, Y_tr, Y_ts):

    # Hyperparameters (that can be tuned later)

    # Stops the tree depth from exploding
    maxDepth = 20          
    # Prevents any splitting tiny groups
    minNode = 5            
    # Number of trees in the forest
    numTrees = 100           
    # How much of the data each tree sees
    dataFraction = 0.7         
    # Only allow each tree to look at sqrt(number of features) when splitting
    # Forces trees to learn differently and prevents same decisions
    featureSubcount = int(np.sqrt(X_tr.shape[1])) 

    # Create and train my Random Forest
    forest = RandomForest(
        numTrees = numTrees,
        maxDepth = maxDepth,
        minNode = minNode,
        dataFraction = dataFraction,
        featureSubcount = featureSubcount
    )
    forest.fit(X_tr, Y_tr)

    # Performs a prediction on the test set
    prediction = forest.predict(X_ts)

    # Computes the accuracy
    correct = np.sum(prediction == Y_ts)
    total = len(Y_ts)
    accuracy = correct / total

    print(f"Random Forest Accuracy = {accuracy:.4f} ({correct}/{total} correct)")

    return prediction



def main(args):
    """
    Perform main logic of program
    """
    # load dataset
    print("Loading dataset ... ")
    X, X_p, X_d, X_c, Y = load_data(args.root)

    # encode labels
    print("Encoding labels ... ")
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    print("Dataset statistics:")
    print("\t Classes: {}".format(len(le.classes_)))
    print("\t Samples: {}".format(len(Y)))
    print("\t Dimensions: ", X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print("Shuffling dataset using seed {} ... ".format(SEED))
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # split
    print("Splitting dataset using train:test ratio of {}:{} ... ".format(int(args.split*10), int((1-args.split)*10)))
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]

    # perform stage 0
    print("Performing Stage 0 classification ... ")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = \
        do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the quantitative attributes processed from the flows
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))

    # perform final classification
    print("Performing Stage 1 classification ... ")
    pred = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts)

    # print classification report
    print(classification_report(Y_ts, pred, target_names=le.classes_))


if __name__ == "__main__":
    # parse cmdline args
    args = parse_args()
    main(args)
