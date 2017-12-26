## Test Network
import mnist_loader as mgpl
import numpy as np

training_data, validation_data, test_data = mgpl.load_data_wrapper()

import network2

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
"""
k = 0.1
upper = 1
ev_a = []
ev_k = []
while k < upper:
	ev_a.append(net.SGD(training_data, 30, 2000, k, evaluation_data=validation_data, monitor_evaluation_accuracy=True))
	ev_k.append(k)
	k+=0.1
#zip(ev_a[])
#print np.maxarg(ev_a[0][2])
max_acc = [max(ev_a[i][1]) for i in range(0,len(ev_a))]
test = zip(ev_k,max_acc)
"""


test = (net.SGD(training_data, 1, 10, 0.5, evaluation_data=validation_data, monitor_evaluation_accuracy=True))
#print test[1]

