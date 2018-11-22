import numpy
import numpy.matlib

class LinearLogisticRegression(object):

    def __init__(self, learningRate = 0.5, maxIterations = 100):
        self.__learningRate = learningRate
        self.__maxIterations = maxIterations
        self.__theta = numpy.array([1, 1, 1])

    def fit(self, X, y):
        b = numpy.ones((X.shape[0], 1))
        self.__X = numpy.concatenate((X,b), axis=1)
        self.__Y = y
        if len(set(self.__Y)) > 2:
            return
        self.__y1 = list(set(self.__Y))[0]
        self.__y2 = list(set(self.__Y))[1]
        mapToZeroOne = lambda x:  0 if x == self.__y1 else  1
        self.__Y = numpy.array([mapToZeroOne(yi) for yi in self.__Y])
        self.__m = len(X)
        self.__theta = self.newtonIteration(self.__maxIterations, self.__theta)

    def gFunc(self, X, theta):
        if numpy.dot(theta, X) >= 0:
            return 1 / (1 + numpy.exp(-numpy.dot(theta, X)))
        else:
            tmp = numpy.exp(numpy.dot(theta, X))
            return tmp / (tmp + 1)

    def gradient(self, theta):
        summe = 0.0
        for i in range(0,self.__m): summe += (self.__Y[i] - self.gFunc(self.__X[i], theta)) * self.__X[i]
        return summe

    def hessianLikeliHood(self, theta):
        summe = 0.0
        for i in range(0, self.__m):
            g = self.gFunc(theta, self.__X[i])
            summe += g * (1 - g) * numpy.dot(self.__X[i], self.__X[i])
        return -summe

    def lFunc(self, theta):
        summe = 0.0
        for i in range(0, self.__m):
            summe += self.__Y[i] * numpy.dot(theta, self.__X[i]) + numpy.log(1 - self.gFunc(self.__X[i], theta))
        return summe
    def newtonIteration(self, iterations, startTheta):
        theta = startTheta
        for i in range(0, iterations):
            print (self.lFunc(theta))
            if numpy.sum(numpy.absolute((self.gradient(theta)))) < 1e-20:
                return theta
            if self.hessianLikeliHood(theta) == 0.0:
                return theta
            theta = theta -  self.__learningRate * 1/self.hessianLikeliHood(theta) * self.gradient(theta)
        return theta

    def predict(self, X):
        b = numpy.ones((X.shape[0],1))
        X = numpy.concatenate((X,b), axis=1)
        indices = numpy.zeros(len(X))
        for i in range (0, len(X)):
            if self.gFunc(X[i], self.__theta) > 0.5:
                indices[i] = self.__y2
            else:
                indices[i] = self.__y1
        return indices


