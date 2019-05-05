
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from collections import Counter
import numpy as np
import os

class Ensemble(object):

    def __init__(self, path, bootstrapAmt, debug=False):
        self.path = path
        self.InputData = np.loadtxt(self.path, skiprows=1, delimiter=",")
        self.bootsrapAmt = bootstrapAmt
        self.debug = debug
        
        # bagging
        self.bootstrap = np.copy(self.InputData)
        np.random.shuffle(self.bootstrap)
        self.bootstrap = self.bootstrap[:self.bootsrapAmt]
    
    def TrainModel(self):
        data = list(zip(*self.bootstrap))    
        self.navieBayesModel = GaussianNB().fit(np.array(data[:len(data)-1]).T, data[len(data)-1])        
        self.regressionModel = LogisticRegression().fit(np.array(data[:len(data)-1]).T, data[len(data)-1])
        self.vectorMachine = SVC().fit(np.array(data[:len(data)-1]).T, data[len(data)-1])
        print("Train OK")

    def Test(self, data):  
        a = np.array(data).reshape(-1, 2)
        nv, reg, svm = int(self.navieBayesModel.predict(a)[0]), int(self.regressionModel.predict(a)[0]),  int(self.vectorMachine.predict(a)[0])
        vote = { "data" : a, "votes": [], "result": -1}
        vote["votes"].append(nv)
        vote["votes"].append(reg)
        vote["votes"].append(svm)
        vote["result"] = Counter(vote["votes"]).most_common(1)[0][0]
        if(self.debug):
            print(vote)
            print("naive bayes result:", nv)
            print("regression result:", reg)
            print("SVM result:", svm)

        return Counter(vote["votes"]).most_common(1)[0][0]
    
        
    def CheckAccuracy(self):
        data = np.copy(self.InputData)
        np.random.shuffle(data)
        test = data[:self.bootsrapAmt]
        
        nvC = 0
        regC = 0
        svmC = 0
        correct = 0
        for i in test:
            a = np.array(i[:len(i)-1]).reshape(-1, 2)
            vote = { "data" : a, "votes": [], "result": -1, "correct" : False}   
            nv, reg, svm = int(self.navieBayesModel.predict(a)[0]), int(self.regressionModel.predict(a)[0]),  int(self.vectorMachine.predict(a)[0])
            vote["votes"].append(nv)
            vote["votes"].append(reg)
            vote["votes"].append(svm)
            vote["result"] = Counter(vote["votes"]).most_common(1)[0][0]
            if(vote["result"] == i[2]): 
                vote["correct"] = True
                correct += 1
            if nv  == i[2]:
                nvC += 1
            if reg == i[2]:
                regC += 1
            if svm == i[2]:
                svmC += 1
            print(vote)
        print("nv ", nvC, " out of ", len(test), float(nvC/len(test)))
        print("reg ", regC, " out of ", len(test), float(regC/len(test)))
        print("svm ", svmC, " out of ", len(test), float(svmC/len(test)))
        print("overall, ", correct,  " out of ", len(test))

    def ClassifyNow(self, testPath, debug=False):
        testData =  np.genfromtxt(testPath, delimiter=',')[1:-1]
        i = 0
        for t in testData:
            testData[i] = list(testData[i])
            res = self.Test(t[:len(t)-1])
            testData[i][2] = res
            print("class of ", t ," is ", int(res))
            i += 1
        textFile = np.asarray(testData)
        np.savetxt("result.csv", textFile, delimiter=",")
        print("Done")


esemble = Ensemble("train.csv", 10, debug=False)
data = list(zip(*esemble.bootstrap))
esemble.TrainModel()
esemble.Test(np.array([23.15,19.05]))
esemble.CheckAccuracy()
esemble.ClassifyNow("test.csv")
