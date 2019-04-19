import  numpy as np
import  time
import  matplotlib.pyplot as plt
time1 = time.clock()
train_X = np.load(r"KNNData\train_X.npy")
train_Y = np.load(r"KNNData\train_Y.npy")
test_X = np.load(r"KNNData\test_X.npy")
test_Y = np.load(r"KNNData\test_Y.npy")
print(train_X.shape)
print(train_Y.shape)
print(train_X[0])
print(train_Y[0])
print(test_X.shape)
print(test_Y.shape)
print(test_X[0])
print(test_Y[0])
time2 =time.clock()
print(time2-time1)
