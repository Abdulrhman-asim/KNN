import operator
from operator import itemgetter

import numpy as np
import pandas as pd

def importdata(path):

    data = pd.read_csv(path,header = None, sep=',').values.tolist()
    return data

def get_classes(data):

    classes = []
    for l in data:
        if l[-1] not in classes:
            classes.append(l[-1])
    return classes

def calc_distance(a,b):

    distance = 0
    for x in range(len(a)-1):
        distance += (b[x] - a[x])**2

    distance = distance**0.5
    return distance

def find_KNneighbors(k,test, trainingSet):

    n = []
    for i in range(len(trainingSet)):
        d = calc_distance(trainingSet[i], test)
        n.append((d,trainingSet[i][-1]))

    n.sort(key=itemgetter(0))

    nearest = pd.DataFrame(n[:k], columns=['distance','classs'])

    return nearest

def choose_class(nearest, trainingSet, Cls):


    tmpClasses = nearest.classs.unique()
    classes = {}
    for x in tmpClasses:
         classes[x] = 0

    for i in range(len(nearest.index)):
        classes[nearest.iloc[i][1]] +=1

    classes = sorted(classes.items(), key=itemgetter(1), reverse=True)

    if len(classes) == 1:
        return classes[0][0]
    if classes[0][1] > classes[1][1]:
        return classes[0][0]

    tied = {}
    for x in range(len(classes)):
        if classes[0][1] == classes[x][1]:
            tied[classes[x][0]] = classes[x][1]

    for x in Cls:
        if x in tied:
            prediction = x
            break

    return prediction




def main():

    trainPath = 'TrainData.csv'
    testPath = 'TestData.csv'

    trainData = importdata(trainPath)
    testData = importdata(testPath)
    classes = get_classes(trainData)
    total = len(testData)
    print(classes)
    for k in range(1, 10):
        print('K value: ', k, '\n')
        correctPreds = 0
        for case in testData:
            candidates = find_KNneighbors(k, case, trainData)
            prediction = choose_class(candidates, trainData, classes)
            print('Predicted Class: ', prediction,' | Actual Class: ', case[-1], '\n')
            if(prediction == case[-1]):
                correctPreds += 1
        print('Correctly classified instances: ', correctPreds,' | Total number of instances: ', total ,'\n')
        print('Accuracy: ', correctPreds / total, '\n', '-------------------------------------------------------', '\n')
        correctPreds = 0






if __name__ == "__main__":
    main()