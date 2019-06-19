import base64
import io
import os
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn import tree
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score,
                                     train_test_split)
from xgboost.sklearn import XGBClassifier

warnings.filterwarnings('ignore')

class GenerateReport():
    def __init__(self, search=False, classifiers=['LR','XG']):
        self.models = {
            'LR': LogisticRegression(solver="sag", n_jobs=-1),
            'RF': RandomForestClassifier(n_jobs=-1),
            'XG': XGBClassifier(n_jobs=-1),
            'DT': tree.DecisionTreeClassifier(max_depth=3),
        }
        self.space = {}
        self.space['XG'] = {
            'max_depth': hp.choice('max_depth', [*range(1,16)]),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
            'n_estimators': hp.choice('n_estimators', [*range(25,275,25)]),
            'min_child_weight': hp.uniform("min_child_weight", 0, 10),
            'max_delta_step': hp.uniform("max_delta_step", 0, 10),
            'subsample': hp.uniform('subsample', 0.8, 1),
            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.8, 1),
            'reg_alpha':  hp.uniform("reg_alpha", 0, 10),
            'reg_lambda': hp.uniform("reg_lambda", 0, 10),
        }
        self.space['LR'] = {
            'C': hp.uniform("C", 0, 10),
            'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear','sag', 'saga']),
        }
        self.space['RF'] = {
            'max_depth': hp.choice('max_depth', [*range(1,16)]),
            'n_estimators': hp.choice('n_estimators', [*range(25,275,25)]),
            'criterion': hp.choice('criterion', ['gini','entropy']),
            'min_samples_split': hp.uniform('min_samples_split', 0, 100), 
            'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 100), 
            'max_features': hp.choice('max_features', ['auto', 'log2', 0.5]),
        }
        self.classifiers = classifiers
        self.search = search
        self.results = []
        self.feature_importance = []
        self.imgbuffer = io.BytesIO()
        
    def fit(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        plt.figure(figsize=(18,7*len(self.classifiers))); i = 1
        for classifier in self.classifiers:
            instance = clone(self.models[classifier])
            self.results.append({
                'Name': classifier,
                'Params': instance.get_params(),
                'Score': cross_val_score(instance, features, labels, cv=4, scoring='roc_auc').mean()
            })
            if self.search:
                objective = partial(self.objective, model=instance, features=features,labels=labels)
                trials = Trials()
                max_evals = self.search if isinstance(self.search, int) else 10
                best = fmin(fn=objective, space=self.space[classifier], trials=trials, algo=tpe.suggest, max_evals=max_evals)
                instance.set_params(**space_eval(self.space[classifier], best))
                self.results.append({
                    'Name': classifier+'_Tuned',
                    'Params': instance.get_params(),
                    'Score': -trials.best_trial['result']['loss'],
                })
            instance.fit(X_train, y_train)
            predict = [i[1] for i in instance.predict_proba(X_test)]
            fpr, tpr, _ = roc_curve(y_test, predict)
            precision, recall, _ = precision_recall_curve(y_test, predict)

            plt.subplot(len(self.classifiers),2,i).plot(fpr, tpr)
            plt.plot([0,1],[0,1],ls='--')
            plt.title(classifier+' Receiver Operating Characteristic Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            i += 1

            plt.subplot(len(self.classifiers),2,i).plot(recall, precision)
            plt.title(classifier+' Precision Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            i += 1

            try:
                features_temp = pd.DataFrame({classifier: features.columns, 'importance': instance.coef_[0] if classifier=='LR' else instance.feature_importances_})
                features_temp = features_temp.sort_values('importance',ascending=False).head(min(20,len(features_temp)))
                features_temp = features_temp.reset_index(drop=True)
                self.feature_importance.append(features_temp)
            except:
                print('no feature name exist')
        plt.savefig(self.imgbuffer,format='png')

    def print(self,filepath):
        if os.path.isfile(filepath):
            if os.path.exists(filepath):
                os.remove(filepath)
            report_path = filepath
        else:
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            report_path = os.path.join(filepath,'report.md')
            
        with open(report_path,'w') as file:
            file.write('## 1 分类器结果\n\n | 分类器 | 参数 | AUC |\n | ------ | ------ | ------ |\n')
            for class_dict in self.results:
                file.write('| '+class_dict['Name']+' | '+str(class_dict['Params'])+' | '+str(class_dict['Score'])+' |\n')
            file.write('\n## 2 评价曲线\n')
            base64_str = base64.b64encode(self.imgbuffer.getvalue())
            file.write('![desc](data:image/png;base64,' + str(base64_str, encoding='utf-8')+')\n')
            file.write('\n## 3 特征描述\n')
            if self.feature_importance!=[]: 
                feature_importance_out = pd.concat(self.feature_importance,axis=1)
                feature_importance_out = pd.concat([pd.DataFrame([['------',]*len(feature_importance_out.columns)], columns=feature_importance_out.columns),feature_importance_out])
                form_out = io.StringIO()
                feature_importance_out.to_csv(form_out, sep="|", index=False)
                file.write(form_out.getvalue()) 
        
    def objective(self, hyperparams, model, features, labels):
        score = cross_val_score(model, features, labels, cv=4, scoring='roc_auc').mean()
        return {'loss': -score, 'status': STATUS_OK}
