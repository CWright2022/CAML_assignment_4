#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time

import tqdm

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), default= "iot_data", 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.8, 
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
    def __init__(self, max_depth = 20, min_node = 5, feature_subcount = None, rng = None):
         # Maximum depth the tree can grow to prevent overfiting
        self.max_depth = max_depth       
        # Smallest number of samples allowed in a node before stopping
        self.min_node = min_node
        # If set, only a random subset of features are considered per split          
        self.feature_subcount = feature_subcount  
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

    def best_split(self, X, y):
        """
        Tries splitting on each feature and threshold
        Selects the split that gives the lowest weighted Gini impurity

        """
        num_samples, num_features = X.shape
        # Sets the parent Gini node to that Gini value prior to any splitting
        parent_gini = self.gini(y)

        # Decide which features to attempt, either all feature or a random subset
        if self.feature_subcount is None:
            features = range(num_features)
        else:
            features = self.rng.choice(num_features, self.feature_subcount, replace = True)

        # Variables intended to store and track the best result
        best_feature, best_threshold = None, None
        best_left_index, best_right_index = None, None
        # Starts with zero improvement
        best_gini = parent_gini  

        # Attempts to conduct splitting on each feature
        for feat in features:
            # Sort the rows of samples by the feature values, whichmakes threshold search easier
            sorted_index = np.argsort(X[:, feat])
            x_sorted = X[sorted_index, feat]

            # Thresholds occur where the feature value switches
            for i in range(1, num_samples):
                if x_sorted[i] == x_sorted[i - 1]:
                    # Skips if no new threshold is found
                    continue
                thr = (x_sorted[i] + x_sorted[i - 1]) / 2

                left_mask = (X[:, feat] <= thr)
                right_mask = ~left_mask
                y_left, y_right = y[left_mask], y[right_mask]

                # Avoids any empty splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # The weighted gini of the split
                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)
                gini_split = (len(y_left) / num_samples) * gini_left + (len(y_right) / num_samples) * gini_right

                # Keeps the best split when the lower gini is optimal
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feat
                    best_threshold = thr
                    best_left_index = np.where(left_mask)[0]
                    best_right_index = np.where(right_mask)[0]

        return best_feature, best_threshold, best_left_index, best_right_index, best_gini, parent_gini

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
        if depth >= self.max_depth or len(y) < self.min_node:
            return node

        feat, thr, left_index, right_index, best_gini, parent_gini = self.best_split(X, y)

        # If there is no helpful split, stop and return to the leaf node
        if feat is None or best_gini >= parent_gini:
            return node

        # Otherwise, store the split and build the left and right subtrees
        node["feat"] = feat
        node["thr"] = thr
        # Uses recursion to build the left and right branches
        node["left"] = self.build(X[left_index], y[left_index], depth + 1)
        node["right"] = self.build(X[right_index], y[right_index], depth + 1)
        return node

    def fit(self, X, y):
        # Ensures that the labels are integers
        y = y.astype(int)
        # Trains the decision tree
        self.root = self.build(X, y, depth = 0)
        return self

    def predict_one(self, x):
        # Predicts one example by using its splits to traverse the tree 
        node = self.root
        if node is None:
            return None
        while node["feat"] is not None:  
            if x[node["feat"]] <= node["thr"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["pred"]

    def predict(self, X):
        # Predict labels for entire dataset
        return np.array([self.predict_one(sample) for sample in X])


class RandomForest:
    """
    Random Forest occurs with multiple decision trees andn then voting
    Each tree sees a random subset of training data (bootstrap sampling) and features (per-split feature subsampling)

    """
    def __init__(self, num_trees = 100, max_depth = 20, min_node = 5, data_fraction = 0.7, feature_subcount = None):
        self.num_trees = num_trees

        self.max_depth = max_depth
        self.min_node = min_node
        self.data_fraction = data_fraction
        self.feature_subcount = feature_subcount
        self.trees = []
        self.rng = np.random.RandomState(0)

    def fit(self, X, y):
        n = len(y)
        bootstrap_size = max(1, int(self.data_fraction * n))

        for _ in range(self.num_trees
):
            # Bootstrap sampling 
            idx = self.rng.choice(n, size = bootstrap_size, replace = True)
            X_sub, y_sub = X[idx], y[idx]

            tree = DecisionTree(
                max_depth = self.max_depth,
                min_node = self.min_node,
                feature_subcount = self.feature_subcount,
                rng = self.rng
            )
            tree.fit(X_sub, y_sub)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # Collect predictions from each tree
        tree_predictions = []
        for tree in self.trees:
            # Retrieve the tree's predictions for all samples
            predictions = tree.predict(X)  
            tree_predictions.append(predictions)
        tree_predictions = np.array(tree_predictions)

        # Selects the most common label per sample
        final = []
        for col in tree_predictions.T:  # Iterates per sample
            final.append(np.argmax(np.bincount(col)))
        return np.array(final)
    
def do_stage_1(X_tr, X_ts, Y_tr, Y_ts, max_depth=20, min_node=5, num_trees=100, data_fraction=0.7):

    # Hyperparameters (that can be tuned later)
     
    # Only allow each tree to look at sqrt(number of features) when splitting
    # Forces trees to learn differently and prevents same decisions
    feature_subcount = int(np.sqrt(X_tr.shape[1])) 

    # Create and train my Random Forest
    forest = RandomForest(
        num_trees = num_trees
,
        max_depth = max_depth,
        min_node = min_node,
        data_fraction = data_fraction,
        feature_subcount = feature_subcount
    )
    forest.fit(X_tr, Y_tr)

    # Performs a prediction on the test set
    prediction = forest.predict(X_ts)

    # Computes the accuracy
    correct = np.sum(prediction == Y_ts)
    total = len(Y_ts)
    accuracy = correct / total

    print(f"Random Forest Accuracy = {accuracy:.4f} ({correct}/{total} correct)")

    # Node we wanna analyze is the left node from the root on the first tree in the forest
    # I just randomly chose this one because why not
    tree = forest.trees[0]

    node = tree.root
    
    # loop to get the deepest node that is not a leaf node
    # while node is not None and node["feat"] is not None:
    #     prev_node = node
    #     node = node['left']
    # node = prev_node 

    # Can manually select the node we want, though
    try:
        node = node['left']['left']
    except TypeError:  # something went wrong with getting a node this way
        print('Uh oh! Some thing went wrong when selecting a node to calculate Gini importance stuff for.')
        node = None

    if node is None:
        return prediction
    # otherwise, do all this stuff, and then return prediction at the end anyway
    # hopefully this works

    # Walk all the training samples through the tree to recalculate some of the values that we need
    def samples_reaching_node(X, root, target_node):
        indices = []
        for i, x in enumerate(X):
            node_ptr = root
            while True:
                if node_ptr is target_node:
                    indices.append(i)
                    break
                if node_ptr["feat"] is None:  # leaf
                    break
                # follow the split
                if x[node_ptr["feat"]] <= node_ptr["thr"]:
                    node_ptr = node_ptr["left"]
                else:
                    node_ptr = node_ptr["right"]
        return indices

    node_indices = samples_reaching_node(X_tr, tree.root, node)
    y_at_node = Y_tr[node_indices]

    # Step 2: compute class counts and Gini impurity
    counts = np.bincount(y_at_node)
    N = len(y_at_node)
    probs = counts / N
    gini = 1 - np.sum(probs ** 2)

    print("\nInformation for this node:")
    print(f"Feature index used for split: {node['feat']}")
    print(f"Threshold: {node['thr']}")
    print(f"# Samples reaching this node: {N}")
    print(f"Class counts: {counts}")
    print(f"Gini impurity: {gini:.6f}")
    print()

    # Do what we just did for the left and right node now!
    left_child = node["left"]
    right_child = node["right"]

    if left_child is None or right_child is None:
        print("Selected node is a leaf or missing children — cannot compute child GinIs.")
        return prediction

    # Get samples reaching the left child
    left_indices = samples_reaching_node(X_tr, tree.root, left_child)
    y_left = Y_tr[left_indices]
    counts_left = np.bincount(y_left)
    N_left = len(y_left)
    probs_left = counts_left / N_left
    gini_left = 1 - np.sum(probs_left ** 2)

    # Get samples reaching the right child
    right_indices = samples_reaching_node(X_tr, tree.root, right_child)
    y_right = Y_tr[right_indices]
    counts_right = np.bincount(y_right)
    N_right = len(y_right)
    probs_right = counts_right / N_right
    gini_right = 1 - np.sum(probs_right ** 2)

    print("\nInformation for LEFT child:")
    print(f"# Samples: {N_left}")
    print(f"Class counts: {counts_left}")
    print(f"Gini impurity (Left): {gini_left:.6f}")
    print()

    print("Information for RIGHT child:")
    print(f"# Samples: {N_right}")
    print(f"Class counts: {counts_right}")
    print(f"Gini impurity (Right): {gini_right:.6f}")
    print()

    return prediction


def evaluate_decision_tree(X_tr, X_ts, Y_tr, Y_ts, max_depth=20, min_node=5, feature_subcount=None):
    """
    Train and evaluate a single DecisionTree, returning prediction, fit and predict times and accuracy.
    """
    # ensure feature subcount is plausible (DecisionTree will use all features if None)
    start_fit = time.time()
    dt = DecisionTree(max_depth = max_depth, min_node = min_node, feature_subcount = feature_subcount)
    dt.fit(X_tr, Y_tr)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    pred = dt.predict(X_ts)
    pred_time = time.time() - start_pred

    acc = accuracy_score(Y_ts, pred)
    print(f"Decision Tree Accuracy = {acc:.4f} ({np.sum(pred == Y_ts)}/{len(Y_ts)} correct)")
    return pred, fit_time, pred_time, acc


def evaluate_random_forest(X_tr, X_ts, Y_tr, Y_ts, max_depth=20, min_node=5, num_trees=100, data_fraction=0.7):
    """
    Train and evaluate the RandomForest implementation, returning prediction, fit and predict times and accuracy.
    """
    featureSubcount = int(np.sqrt(X_tr.shape[1]))
    start_fit = time.time()
    forest = RandomForest(
        num_trees = num_trees,
        max_depth = max_depth,
        min_node = min_node,
        data_fraction = data_fraction
    )
    forest.fit(X_tr, Y_tr)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    pred = forest.predict(X_ts)
    pred_time = time.time() - start_pred

    acc = accuracy_score(Y_ts, pred)
    print(f"Random Forest Accuracy = {acc:.4f} ({np.sum(pred == Y_ts)}/{len(Y_ts)} correct)")
    return pred, fit_time, pred_time, acc


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, filename=None):
    """
    Generate and display a confusion matrix heatmap.
    
    Parameters
    ----------
    y_true : numpy array
             True labels.
    y_pred : numpy array
             Predicted labels.
    class_names : array-like
                  Names of classes.
    model_name : str
                 Name of the model (for title and filename).
    filename : str, optional
               Path to save the figure. If None, uses model_name to generate filename.
    
    Returns
    -------
    cm : numpy array
         Confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size':6},
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if filename is None:
        filename = f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {filename}")
    plt.close()
    
    return cm


def print_confusion_matrix(cm, class_names, model_name):
    """
    Print confusion matrix in a readable format.
    
    Parameters
    ----------
    cm : numpy array
         Confusion matrix.
    class_names : array-like
                  Names of classes.
    model_name : str
                 Name of the model.
    """
    print(f"\nConfusion Matrix - {model_name}:")
    print("=" * 80)
    
    # Print header with class names
    header = "Predicted →"
    for class_name in class_names:
        header += f"\t{class_name[:10]}"
    print(header)
    print("-" * 80)
    
    # Print each row
    for i, class_name in enumerate(class_names):
        row = f"{class_name[:10]}\t"
        for j in range(len(class_names)):
            row += f"\t{cm[i][j]}"
        print(row)
    
    print("=" * 80)



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
    print("\t Samples: {}".format(np.asarray(Y).shape[0]))
    print("\t Dimensions: ", X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print("Shuffling dataset using seed {} ... ".format(SEED))
    Y = np.asarray(Y)
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
    max_depth=15
    min_node=7
    num_trees=200
    data_fraction=1.2
    print(f"Testing with maxDepth={max_depth}, min_node={min_node}, num_trees={num_trees}, data_fraction={data_fraction}")
    # Evaluate single Decision Tree (non-ensemble)
    featureSubcount_dt = int(np.sqrt(X_tr_full.shape[1]))
    time_start = time.time()
    dt_pred, dt_fit_time, dt_pred_time, dt_acc = evaluate_decision_tree(
        X_tr_full, X_ts_full, Y_tr, Y_ts,
        max_depth=max_depth, min_node=min_node
    )
    time_end = time.time()
    print(f"Decision Tree evaluation time: {time_end - time_start:.4f}s")

    time_start=time.time()
    # Evaluate Random Forest (ensemble)
    rf_pred, rf_fit_time, rf_pred_time, rf_acc = evaluate_random_forest(
        X_tr_full, X_ts_full, Y_tr, Y_ts,
        max_depth=max_depth, min_node=min_node, num_trees=num_trees
, data_fraction=data_fraction
    )
    time_end = time.time()
    print(f"Random Forest evaluation time: {time_end - time_start:.4f}s")

    # Summary of results
    print("\nSummary:\n")
    print(f"Decision Tree  - fit: {dt_fit_time:.4f}s, predict: {dt_pred_time:.4f}s, accuracy: {dt_acc:.4f}")
    print(f"Random Forest  - fit: {rf_fit_time:.4f}s, predict: {rf_pred_time:.4f}s, accuracy: {rf_acc:.4f}")

    print("\nClassification report for Decision Tree:\n")
    print(classification_report(Y_ts, dt_pred, target_names=le.classes_))

    print("\nClassification report for Random Forest:\n")
    print(classification_report(Y_ts, rf_pred, target_names=le.classes_))

    # Generate and display confusion matrices
    print("\n" + "=" * 80)
    print("GENERATING CONFUSION MATRICES")
    print("=" * 80)
    
    # Decision Tree confusion matrix
    dt_cm = plot_confusion_matrix(Y_ts, dt_pred, le.classes_, "Decision Tree", 
                                   filename="confusion_matrix_decision_tree.png")
    print_confusion_matrix(dt_cm, le.classes_, "Decision Tree")
    
    # Random Forest confusion matrix
    rf_cm = plot_confusion_matrix(Y_ts, rf_pred, le.classes_, "Random Forest",
                                   filename="confusion_matrix_random_forest.png")
    print_confusion_matrix(rf_cm, le.classes_, "Random Forest")


if __name__ == "__main__":
    # parse cmdline args
    args = parse_args()
    main(args)
