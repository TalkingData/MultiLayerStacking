import numpy
from mlstacking.sklearn import StackingModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

X = numpy.random.rand(10,10)
Y = numpy.random.randint(0,2,(10,1))

base_models = [[DecisionTreeClassifier(),RandomForestClassifier(),],
                [RandomForestClassifier(),XGBClassifier(),],
                [XGBClassifier(),DecisionTreeClassifier(),],]

sm = StackingModel(base_models,XGBClassifier())
sm.fit(X,Y)

sm.predict(X)
# array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1])

sm.predict_proba(X)
# array([[0.6039953 , 0.39600468],
#        [0.6039953 , 0.39600468],
#        [0.6039953 , 0.39600468],
#        [0.40033996, 0.59966004],
#        [0.6039953 , 0.39600468],
#        [0.6039953 , 0.39600468],
#        [0.6039953 , 0.39600468],
#        [0.40033996, 0.59966004],
#        [0.40033996, 0.59966004],
#        [0.40033996, 0.59966004]], dtype=float32)
