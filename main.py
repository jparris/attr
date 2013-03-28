import numpy as np
from scipy import io
from libsvm.svmutil import *
from sklearn import cross_validation
import random
from sklearn.svm import SVR

def normalize(l):
    a = np.array(l)
    return list((a - np.mean(a))/np.std(a))
    

def weights(binary, ordering):
    index = 0
    curr = binary[ordering[index]]
    
    # Find The break where the odering shifts from neg to pos attribute
    while index < len(ordering)-1:
        if curr == binary[ordering[index + 1]]:
            index += 1
            curr == binary[ordering[index]]
        else:
            break
    

    # Assign Values for Positive Side
    pos = []
    const = 1.0/ (len(ordering) - index)
    i = const 
    for j in range(index + 1, len(ordering)):
        pos.append(i)
        i+=const

    # Assign Values for Negative Side
    neg = []
    const = -1.0/ (index+1)
    i = const
    r = range(0, index + 1)
    r.reverse()
    for j in r:
        neg.insert(0, i)
        i += const
    
    l = neg + pos # cat lists together

    wieghts = {}
    for i in range(len(l)):
        wieghts[binary[ordering[i]]] = l[i]

    return wieghts

 
def subSet(X, y, labels, wieghts):
    data = []
    ls = []
    for label in  labels:
        i = 0
        for target in y:
            if label in target:
                data.append(list(X[i]))
                ls.append(wieghts[label])
            i += 1
    return data, ls
        
def canary(X, y, wieghts):
    ys = []
    for row in y:
        ys.append(row[0][0 : row[0].find('_')]) #
    classes = list(set(ys)) 
    
    c = random.sample(classes, 4)
    X_train, y_train = subSet(X, y, c, wieghts)
    y_test = ys
    X_test = X

    print y_train
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    y_rbf = svr_rbf.fit(X_train, y_train).fit(X)
#    prob = svm_problem(y_train, X_train)
#    param = svm_parameter('-s 3 -t 2')
#    m = svm_train(prob, param) 

#    y, x = svm_read_problem('libsvm/heart_scale')
#    m = svm_train(y[:200], x[:200], '-c 4')
#    p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
#    print p_acc

#    #    #m = svm_train(prob, param)
    return

def main():
    #Load the data
    rawData = io.loadmat("./data/pubfig.mat")
    features = rawData['feat']
    labels = rawData['im_names'].reshape(-1)

    #TODO Split into train vs testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        features, labels, test_size=0.33, random_state=0)

    #TODO Assign weights to classes
    w = weights({'A':1, 'C': 1, 'H': 1, "J": 1, "M": 0, "S": 0, "V": 1, "Z": 1}, "S M Z V J A H C".split(" "))
    print w
    #TODO Run Canary Test
    canary(features, labels, w)

if __name__ == "__main__":
    main()
