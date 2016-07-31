# -*-coding:utf-8-*-
import csv
from numpy import *
import cmath
import operator

def loadTrainData():
    l = []
    with open("train.csv") as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = array(l)
    label = l[:,0]
    data = l[:,1:]
    return nomalizing(toInt(data)), toInt(label)

def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    data = array(l)
    return nomalizing(toInt(data))


def toInt(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j] = int(array[i,j])

    return newArray

def nomalizing(array):
    m, n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j] != 0:
                array[i,j] = 1

    return array

def saveResult(result):
    with open('result.csv', 'wb') as myFile:
        myWrite = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWrite.writerow(tmp)

from sklearn import svm

def svcClassify(trainData, trainLabel, testData):
    svcClf = svm.SVC(C = 5.0)
    svcClf.fit(trainData, ravel(trainLabel))
    testLabel = svcClf.predict(testData)
    saveResult(testLabel)

def recognition():
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    svcClassify(trainData,trainLabel, testData)

recognition()