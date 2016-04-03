#!/usr/bin/env python
# -*- coding: utf-8 -*-
from kNN import *
from numpy import *

# test1: AABB k=3
group, labels = kNN.createDataSet()
print(kNN.classify0([-2, 1.1], group, labels, 3))

# test2: Dating
datingDataMat, datingLabels = kNN.file2matrix("datingTestSet.txt")

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
           15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

datingClassTest()