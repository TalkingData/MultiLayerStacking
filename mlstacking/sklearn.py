# coding = UTF-8
# Author: Hao Wang <hao.wang2@tendcloud.com>
# License: BSD 3-Clause License

"""Scikit-Learn Wrapper interface for multi-layer stacking."""
from __future__ import absolute_import
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.utils import check_X_y, check_array
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
import numpy as np

class StackingModel(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Implementation of the Scikit-Learn API for multi-layer stacking.

    Parameters
    ----------
    base_models : list
        List of list of sklearn type classifiers
    meta_model : object
        Sklearn type classifiers
    predict_mode : string
        Specify which predict to use: average, once
    n_folds : int
        Depend how many folds each classifier run
    keep_layer_results : boolean
        Keep results of each layer or not
    """

    def __init__(self, base_models, meta_model, predict_mode='average',  n_folds=5, keep_layer_results=True):
        if not hasattr(base_models[0] , '__iter__'):
            raise AttributeError("object in 'base_models' is not iterable")

        if predict_mode!='average' and predict_mode!='once':
            raise AttributeError("'predict_mode' can't be '{}'".format(str(predict_mode)))
        else:
            self.predict_mode = predict_mode

        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.keep_layer_results = keep_layer_results

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csc', 'csc'])
        self.base_models_ = [[list()for x in y] for y in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        self.train_layer_results_ = []

        for layer_num in range(len(self.base_models)):
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            predictions = np.zeros((X.shape[0], len(self.base_models[layer_num])))
            for i, model in enumerate(self.base_models[layer_num]):
                if self.predict_mode=='average':
                    for train_index, predict_index in kfold.split(X, y):
                        instance = clone(model)
                        instance.fit(X[train_index], y[train_index])
                        self.base_models_[layer_num][i].append(instance)
                        predictions[predict_index, i] = [proba[-1] for proba in instance.predict_proba(X[predict_index])]
                elif self.predict_mode=='once':
                    instance = clone(model)
                    instance.fit(X, y)
                    self.base_models_[layer_num][i].append(instance)
                    predictions[:, i] = [proba[-1] for proba in cross_val_predict(instance, X, y, cv=self.n_folds, n_jobs=-1, method='predict_proba')]
            X = predictions
            if self.keep_layer_results:
                self.train_layer_results_.append(X)
        self.meta_model_.fit(X, y)
        return self

    def predict_layer_features(self, X, layer):       
        return np.column_stack([np.column_stack([[proba[-1] for proba in model.predict_proba(X)] for model in base_models]).mean(axis=1) for base_models in layer])
        
    def predict_layers_features(self, X):
        for layer in self.base_models_:
            X = self.predict_layer_features(X, layer)
            if self.keep_layer_results:
                self.train_layer_results_.append(X)
        return X

    def predict(self, X):
        X = self.predict_layers_features(check_array(X, accept_sparse='csc'))
        return self.meta_model_.predict(X)

    def predict_proba(self, X):
        X = self.predict_layers_features(check_array(X, accept_sparse='csc'))
        return self.meta_model_.predict_proba(X)
