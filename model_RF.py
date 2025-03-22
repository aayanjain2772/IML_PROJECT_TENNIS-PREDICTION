import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import mode

# Simple Decision Tree using sklearn-like API
class SimpleDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
    
    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        # Stop conditions
        if (depth >= self.max_depth or n_samples < self.min_samples_split 
                or self.n_classes == 1):
            self.leaf_value = mode(y).mode[0]
            return
        
        # Select random subset of features
        if self.max_features is not None:
            feats_idx = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feats_idx = np.arange(n_features)
        
        # Find best split
        best_feat, best_thresh, best_gain = None, None, -1
        for feat in feats_idx:
            thresholds = np.unique(X[:, feat])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feat], threshold)
                if gain > best_gain:
                    best_feat, best_thresh, best_gain = feat, threshold, gain
        
        if best_gain == -1:
            self.leaf_value = mode(y).mode[0]
            return
        
        # Store best split parameters
        self.feat_idx = best_feat
        self.threshold = best_thresh

        # Split data
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh

        # Recursive building
        self.left = SimpleDecisionTree(
            max_depth=self.max_depth, 
            min_samples_split=self.min_samples_split,
            max_features=self.max_features
        )
        self.left.fit(X[left_idxs], y[left_idxs], depth+1)

        self.right = SimpleDecisionTree(
            max_depth=self.max_depth, 
            min_samples_split=self.min_samples_split,
            max_features=self.max_features
        )
        self.right.fit(X[right_idxs], y[right_idxs], depth+1)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        if hasattr(self, 'leaf_value'):
            return self.leaf_value
        if inputs[self.feat_idx] <= self.threshold:
            return self.left._predict(inputs)
        else:
            return self.right._predict(inputs)
    
    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _information_gain(self, y, X_col, split_thresh):
        parent_entropy = self._entropy(y)

        left_idxs = X_col <= split_thresh
        right_idxs = X_col > split_thresh
        
        if sum(left_idxs) == 0 or sum(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = sum(left_idxs), sum(right_idxs)

        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        return parent_entropy - child_entropy


# Random Forest classifier built from scratch
class CustomRandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples, n_features = X.shape

        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features

        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = SimpleDecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return mode(tree_preds, axis=0).mode[0]

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
