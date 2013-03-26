from scipy import io
from libsvm.svmutil import *
from sklearn import cross_validation


def weights(binary, ordering):
    index = 0
    curr = binary[ordering[index]]
    while index < len(ordering)-1:
        if curr == binary[ordering[index + 1]]:
            index += 1
            curr == binary[ordering[index]]
        else:
            break
    print index

def canary():
#    y, x = svm_read_problem('libsvm/heart_scale')
#    m = svm_train(y[:200], x[:200], '-c 4')
#    p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
#    print p_acc
#prob  = svm_problem(y, x)
    #param = svm_parameter('-t 4 -c 4 -b 1')
    #m = svm_train(prob, param)
    return Train, Test


def main():
    #Load the data
    rawData = io.loadmat("./data/pubfig.mat")
    features = rawData['feat']
    labels = rawData['im_names'].reshape(-1)

    #TODO Split into train vs testing
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                        features, labels, test_size=0.33, random_state=0)

    #TODO Assign weights to classes
    weights({'A':1, 'C': 1, 'H': 1, "J": 1, "M": 0, "S": 0, "V": 1, "Z": 1}, "S M Z V J A H C".split(" "))
    #TODO Run Canary Test


if __name__ == "__main__":
    main()
