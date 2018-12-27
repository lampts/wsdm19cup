import numpy as np

print("(*) concatenate train into 64-d")
x1 = np.loadtxt("../data/X_33_norm.npy")
x2 = np.loadtxt("../data/Xmin_norm.npy")
x = np.hstack((x1,x2))
np.savetxt("../data/X_64.npy", x)

print("(*) concatenate test into 64-d") 
x1 = np.loadtxt("../data/X_test_33_norm.npy")
x2 = np.loadtxt("../data/Xmin_test_norm.npy")
x = np.hstack((x1,x2))
np.savetxt("../data/X_test_64.npy", x)
