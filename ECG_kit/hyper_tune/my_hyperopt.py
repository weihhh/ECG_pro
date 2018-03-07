from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print(accuracy_score(y_test, y_pred))

def percept(args):
    global X_train_std,y_train,y_test
    ppn = Perceptron(n_iter=int(args["n_iter"]),eta0=args["eta"]*0.01,random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    return -accuracy_score(y_test, y_pred)

# from hyperopt import fmin,tpe,hp,partial
# space = {"n_iter":hp.choice("n_iter",[i for i in range(30,50)]),
#          "eta":hp.uniform("eta",0.05,0.5)}
# algo = partial(tpe.suggest,n_startup_jobs=10)
# best = fmin(percept,space,algo = algo,max_evals=10)
# print(best)
# print(percept(best))