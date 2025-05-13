import pandas as pd
import numpy as np
from collections import defaultdict

class my_NB:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.priors = {} 
        self.likelihoods = defaultdict(lambda: defaultdict(dict))  
        self.classes_ = [] 

    def fit(self, X, y):
        self.classes_ = sorted(list(set(y))) 
        total_count = len(y)
        
        for c in self.classes_:
            self.priors[c] = (np.sum(y == c) + self.alpha) / (total_count + len(self.classes_) * self.alpha)

        # Calculate likelihoods P(X|Y) for each feature
        for feature in X.columns:
            for c in self.classes_:
                feature_counts = X[y == c][feature].value_counts()
                for category, count in feature_counts.items():
                    self.likelihoods[feature][c][category] = (count + self.alpha) / (np.sum(y == c) + len(feature_counts) * self.alpha)

    def predict(self, X):
        predictions = []
        for _, x in X.iterrows():
            post_probs = {}
            for c in self.classes_:
                post_prob = self.priors[c]
                for feature in X.columns:
                    category = x[feature]
                    # Use likelihoods calculated in fit
                    if category in self.likelihoods[feature][c]:
                        post_prob *= self.likelihoods[feature][c][category]
                    else:
                        post_prob *= (self.alpha / (np.sum(y == c) + len(self.likelihoods[feature][c]) * self.alpha))
                post_probs[c] = post_prob
            
            predictions.append(max(post_probs, key=post_probs.get))
        return predictions

    def predict_proba(self, X):
        prob_list = []
        for _, x in X.iterrows():
            post_probs = {}
            for c in self.classes_:
                post_prob = self.priors[c]
                for feature in X.columns:
                    category = x[feature]
                    if category in self.likelihoods[feature][c]:
                        post_prob *= self.likelihoods[feature][c][category]
                    else:
                        post_prob *= (self.alpha / (np.sum(y == c) + len(self.likelihoods[feature][c]) * self.alpha))
                post_probs[c] = post_prob

            total_prob = sum(post_probs.values())
            normalized_probs = {c: prob / total_prob for c, prob in post_probs.items()}
            prob_list.append(normalized_probs)

        return pd.DataFrame(prob_list, columns=self.classes_)
