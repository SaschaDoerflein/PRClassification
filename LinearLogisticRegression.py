#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de


#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software: 
#    you can redistribute it and/or modify it under the terms of the 
#    GNU General Public License as published by the Free Software Foundation, 
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that 
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.  
#    If not, see <http://www.gnu.org/licenses/>.


import numpy
import numpy.matlib


class LinearLogisticRegression(object):

    # learningRate is the step length for Newton's method
    # maxIterations is the maximum number of iterations for learning the parameters
    def __init__(self, learningRate = 0.5, maxIterations = 100):
        self.__LRate = learningRate
        self.__maxIterations = maxIterations
        self.__t = numpy.array([1, 1])

    def fit(self, X, y):
        self.__X = X
        self.__m = len(X)
        self.__Y = y
        self.__t = self.NRSteps(self.__maxIterations, self.__t)

    def gFunc(self, X, t):
        return numpy.reciprocal(1 + numpy.exp(-numpy.dot(t, X)))

    def NRSteps(self, iterations, t):
        for i in range(0, iterations):
            t -= numpy.reciprocal(self.hessian(t + self.__LRate) * self.gradient(t + self.__LRate))
        return t

    def gradient(self, t):
        sum = 0
        for i in range(0, self.__m): sum += (self.__Y[i] - self.gFunc(numpy.dot(t, self.__X[i]))) * self.__X[i]
        return sum

    def hessian(self, t):
        summe = 0
        for i in range(0, self.__m):
            g = self.gFunc(t, self.__X[i])
            summe += g * (1 - g) * numpy.dot(self.__X[i], self.__X[i])
        return -summe

    def predict(self, X):
        if self.gFunc(X, self.__t) >= 0.5:
            return 1
        else:
            return 0



