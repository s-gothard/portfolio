import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    def __init__(self, predictions, actuals, pred_proba=None):
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if isinstance(self.pred_proba, pd.DataFrame):
            self.classes_ = list(self.pred_proba.columns)
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        correct = self.predictions == self.actuals
        self.acc = float(Counter(correct)[True]) / len(correct)
        self.confusion_matrix = {}

        for label in self.classes_:
            tp = np.sum((self.predictions == label) & (self.actuals == label))
            fp = np.sum((self.predictions == label) & (self.actuals != label))  
            fn = np.sum((self.predictions != label) & (self.actuals == label)) 
            tn = np.sum((self.predictions != label) & (self.actuals != label)) 
            
            self.confusion_matrix[label] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    def precision(self, target=None, average="macro"):
        if self.confusion_matrix is None:
            self.confusion()
        
        if target is None:
            if average == "macro":
                total_precision = [self.precision(label) for label in self.classes_]
                return np.mean(total_precision)
            elif average == "micro":
                total_tp = sum(self.confusion_matrix[label]["TP"] for label in self.classes_)
                total_fp = sum(self.confusion_matrix[label]["FP"] for label in self.classes_)
                if total_tp + total_fp > 0:
                    return total_tp / (total_tp + total_fp)
                else:
                    return 0.0
            elif average == "weighted":
                total_count = sum(self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FP"] for label in self.classes_)
                weighted_precision = 0.0
                
                for label in self.classes_:
                    label_precision = self.precision(label)
                    label_count = self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FP"]
                    weighted_precision += (label_precision * label_count) / total_count
                
                if total_count > 0:
                    return weighted_precision
                else:
                    return 0.0
        else:
            tp = self.confusion_matrix[target]["TP"]
            fp = self.confusion_matrix[target]["FP"]
            if tp + fp > 0:
                return tp / (tp + fp)
            else:
                return 0.0

    def recall(self, target=None, average="macro"):
        if self.confusion_matrix is None:
            self.confusion()
        
        if target is None:
            if average == "macro":
                total_recall = [self.recall(label) for label in self.classes_]
                return np.mean(total_recall)
            elif average == "micro":
                total_tp = sum(self.confusion_matrix[label]["TP"] for label in self.classes_)
                total_fn = sum(self.confusion_matrix[label]["FN"] for label in self.classes_)
                if total_tp + total_fn > 0:
                    return total_tp / (total_tp + total_fn)
                else:
                    return 0.0
            elif average == "weighted":
                total = sum(self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FN"] for label in self.classes_)
                weighted_sum = 0.0
                
                for label in self.classes_:
                    label_recall = self.recall(label)
                    label_count = self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FN"]
                    weighted_sum += label_recall * label_count
                
                if total > 0:
                    return weighted_sum / total
                else:
                    return 0.0
        else:
            tp = self.confusion_matrix[target]["TP"]
            fn = self.confusion_matrix[target]["FN"]
            if tp + fn > 0:
                return tp / (tp + fn)
            else:
                return 0.0

    def f1(self, target=None, average="macro"):
        if self.confusion_matrix is None:
            self.confusion()
        
        if target is None:
            if average == "macro":
                total_f1 = [self.f1(label) for label in self.classes_]
                return np.mean(total_f1)
            elif average == "micro":
                precision_micro = self.precision(target=None, average="micro")
                recall_micro = self.recall(target=None, average="micro")
                if precision_micro + recall_micro > 0:
                    return 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
                else:
                    return 0.0
            elif average == "weighted":
                total = sum(self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FP"] + self.confusion_matrix[label]["FN"] for label in self.classes_)
                weighted_sum = 0.0
                
                for label in self.classes_:
                    f1_value = self.f1(label)
                    label_count = self.confusion_matrix[label]["TP"] + self.confusion_matrix[label]["FP"] + self.confusion_matrix[label]["FN"]
                    weighted_sum += f1_value * label_count
                
                if total > 0:
                    return weighted_sum / total
                else:
                    return 0.0
        else:
            precision_value = self.precision(target)
            recall_value = self.recall(target)
            if precision_value + recall_value > 0:
                return 2 * (precision_value * recall_value) / (precision_value + recall_value)
            else:
                return 0.0
