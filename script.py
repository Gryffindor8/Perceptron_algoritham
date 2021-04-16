import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from pylab import rand, plot, norm
from pylab import show


class Perceptron:
    # initializing the perception
    def __init__(self):
        self.w = rand(2) * 2 - 1
        self.learningRate = 0.1

    def response(self, x):
        # Perception output
        y = x[0] * self.w[0] + x[1] * self.w[1]
        if y >= 0:
            return 1
        else:
            return -1

    def updateWeights(self, x, iterError):
        """
        w(t+1) = w(t) + learningRate * (d-r) * x where d is desired output, r is the perceptron
        response and (d-r) is the iteration error
        """
        self.w[0] += self.learningRate * iterError * x[0]
        self.w[1] += self.learningRate * iterError * x[1]

    def train(self, data):
        """
         trains all the vector in data.
         Every vector in data must have three elements,
         the third element (x[2]) must be the label (desired output)
        """
        learned = False
        iteration = 0
        while not learned:
            globalError = 0.0
            for x in data:
                r = self.response(x)
                if x[2] != r:
                    iterError = x[2] - r
                    self.updateWeights(x, iterError)
                    globalError += abs(iterError)
                iteration += 1
                if globalError == 0.0 or iteration >= 100:
                    print(
                        'iterations', iteration)
                    learned = True

    @staticmethod
    def generate10D(n):
        # Generating a 10D data-set linearly separable
        data = pd.DataFrame(np.random.randn(n, 10),
                            columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
        x1 = (rand(n) * 2 - 1) / 2 - 0.5
        y1 = (rand(n) * 2 - 1) / 2 + 0.5
        x2 = (rand(n) * 2 - 1) / 2 + 0.5
        y2 = (rand(n) * 2 - 1) / 2 - 0.5
        x3 = (rand(n) * 2 - 1) / 2 + 0.5
        y3 = (rand(n) * 2 - 1) / 2 + 0.5
        x4 = (rand(n) * 2 - 1) / 2 - 0.5
        y4 = (rand(n) * 2 - 1) / 2 + 0.5
        x5 = (rand(n) * 2 - 1) / 2 + 0.5
        y5 = (rand(n) * 2 - 1) / 2 - 0.5
        x6 = (rand(n) * 2 - 1) / 2 + 0.5
        y6 = (rand(n) * 2 - 1) / 2 + 0.5
        x7 = (rand(n) * 2 - 1) / 2 - 0.5
        y7 = (rand(n) * 2 - 1) / 2 + 0.5
        x8 = (rand(n) * 2 - 1) / 2 + 0.5
        y8 = (rand(n) * 2 - 1) / 2 - 0.5
        x9 = (rand(n) * 2 - 1) / 2 + 0.5
        y9 = (rand(n) * 2 - 1) / 2 + 0.5
        x10 = (rand(n) * 2 - 1) / 2 - 0.5
        y10 = (rand(n) * 2 - 1) / 2 + 0.5
        inputs = []
        for i in range(len(x1)):
            inputs.append([x1[i], y1[i], 1])
            inputs.append([x2[i], y2[i], -1])
            inputs.append([x3[i], y3[i], 1])
            inputs.append([x4[i], y4[i], -1])
            inputs.append([x5[i], y5[i], 1])
            inputs.append([x6[i], y6[i], -1])
            inputs.append([x7[i], y7[i], 1])
            inputs.append([x8[i], y8[i], -1])
            inputs.append([x9[i], y9[i], 1])
            inputs.append([x10[i],y10[i], -1])
        # return inputs
        return data

    @staticmethod
    def generateData(n):
        # Generating a 2D data-set linearly separable
        xb = (rand(n) * 2 - 1) / 2 - 0.5
        yb = (rand(n) * 2 - 1) / 2 + 0.5
        xr = (rand(n) * 2 - 1) / 2 + 0.5
        yr = (rand(n) * 2 - 1) / 2 - 0.5

        inputs = []
        for i in range(len(xb)):
            inputs.append([xb[i], yb[i], 1])
            inputs.append([xr[i], yr[i], -1])

        return inputs

    @staticmethod
    def main():

        trainset = Perceptron.generateData(100)
        d1 = Perceptron.generate10D(100)
        perceptron = Perceptron()
        perceptron.train(trainset)
        testset = Perceptron.generateData(500)
        for x in testset:
            r = perceptron.response(x)
            if r != x[2]:
                print('error')
            if r == 1:
                plot(x[0], x[1], 'ob')
            else:
                plot(x[0], x[1], 'or')

        # plot of the separation line, which is orthogonal to w
        n = norm(perceptron.w)
        ww = perceptron.w / n
        ww1 = [ww[1], -ww[0]]
        ww2 = [-ww[1], ww[0]]
        # plot(d1,'or')
        plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')
        # plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '--k')

        scatter_matrix(d1, alpha=0.2, figsize=(7, 7))
        # scatter_matrix([ww1[0], ww2[0]], [ww1[1], ww2[1]])
        show()


if __name__ == '__main__':
    Perceptron.main()
