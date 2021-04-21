#!/usr/bin/env python
##
## Read data points for training and test sets, and corresponding
## labels, from comma separated files. Then construct the kernel
## matrix for the training data points. Perform kernel ridge
## regression.
##
## Make sure to compile strumpack as a shared library:
##    add -DBUILD_SHARED_LIBS=ON to the cmake invocation
## then set the LD_LIBRARY_PATH to the install/lib folder where
## libstrumpack is installed
##
## Add CMAKE_INSTALL_PREFIX/lib/ to your LD_LIBRARY_PATH
##
## Add CMAKE_INSTALL_PREFIX/include/python/ to your PYTHONPATH
##

import sys
import numpy as np
import STRUMPACKKernel as sp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("""\
    Usage: OMP_NUM_THREADS=1 mpirun -n 4 python3 KernelRegression.py filename h lambda degree
       - 'filename' should refer to 4 files:
           filename_train.csv
           filename_train_label.csv
           filename_test.csv
           filename_test_label.csv
       - h: kernel width
       - lambda: regularization parameter
       - degree: ANOVA kernel degree
\
    """)

# parse input parameters
fname = './data/susy_10Kn'
h = 1.3
lam = 3.11
degree = 1
nargs = len(sys.argv)
if (nargs > 1):
    fname = sys.argv[1]
if (nargs > 2):
    h = float(sys.argv[2])
if (nargs > 3):
    lam = float(sys.argv[3])
if (nargs > 4):
    degree = float(sys.argv[4])

# read data
prec = np.float64
train_points = np.genfromtxt(
    fname + '_train.csv', delimiter=",", dtype=prec
)
train_labels = np.genfromtxt(
    fname + '_train_label.csv', delimiter=",", dtype=prec
)
test_points = np.genfromtxt(
    fname + '_test.csv', delimiter=",", dtype=prec
)
test_labels = np.genfromtxt(
    fname + '_test_label.csv', delimiter=",", dtype=prec
)
n, d = train_points.shape
m = test_points.shape[0]
if rank == 0:
    print('n =', n, 'd =', d, 'm =', m)


def quality(p, l):
    return 100.*(m - sum(p[i]*l[i] < 0 for i in range(m))) / m


# Kernel ridge regression classification
# using HSS approximation of the kernel
K_HSS = sp.STRUMPACKKernel(
    h, lam, degree, kernel='rbf', approximation='HSS', mpi=True, argv=sys.argv)
K_HSS.fit(train_points, train_labels)
pred = K_HSS.predict(test_points)
# check quality, labels are -1 or +1

if rank == 0:
    print('HSS KernelRR quality =', quality(pred, test_labels), '%')
    print("classes:", K_HSS.classes_)


# read these again because they were permuted, which shouldn't really
# be an issue
train_points = np.genfromtxt(fname + '_train.csv', delimiter=",")
train_labels = np.genfromtxt(fname + '_train_label.csv', delimiter=",")

# Kernel ridge regression classification
# using HODLR approximation of the kernel
K_HODLR = sp.STRUMPACKKernel(
    h, lam, kernel='rbf', approximation='HODLR', mpi=True, argv=sys.argv)
K_HODLR.fit(train_points, train_labels)
pred = K_HODLR.predict(test_points)
# check quality, labels are -1 or +1
if rank == 0:
    print('HODLR KernelRR quality =', quality(pred, test_labels), '%')
    print("classes:", K_HODLR.classes_)


# # grid-search for hyper-parameters
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# # grid_list = {"h": np.arange(0.1, 5.0, 0.5),
# #              "lam": np.arange(1.0, 20.0, 5.0)}
# ## 10^-2 -> 10^2 in num=5 steps
# grid_list = {"h": np.logspace(-2, 2, num=3),
#              "lam": np.logspace(-2, 2, num=3)}
# grid_search = GridSearchCV(K, param_grid=grid_list, cv=3)
# grid_search.fit(train_points, train_labels.round())
# # print(grid_search.cv_results_)
# if rank == 0:
#     print("best_params_ =", grid_search.best_params_)
# K = grid_search.best_estimator_
# pred_gs = grid_search.predict(test_points)
# # check quality, labels are -1 or +1
# if rank == 0:
#     print('HSS KernelRR grid search quality =',
#           quality(pred_gs, test_labels), '%')


# # randomized search for hyperparameters
# #...


# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
# import matplotlib.pyplot as plt

# pred_train = cross_val_predict(K, train_points, train_labels, cv=3)
# if rank == 0:
#     print("# precision score = TP / (TP + FP)")
#     print("#    recall score = TP / (TP + FN)")
#     print("#  f1 score = 2 / (1/precision + 1/recall)"
#           " = TP / (TP + (FN+FP)/2)")
#     print("  precision score =", precision_score(train_labels, pred_train))
#     print("     recall score =", recall_score(train_labels, pred_train))
#     print("         f1 score =", f1_score(train_labels, pred_train))


# scores = cross_val_predict(
#     K, train_points, train_labels, cv=3, method="decision_function")

# plt.figure(0)
# precisions, recalls, thresholds = precision_recall_curve(train_labels, scores)
# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
#     plt.xlabel("Threshold")
#     plt.legend(loc="center right")
#     plt.ylim([0, 1])
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

# plt.figure(1)
# ## roc curve: reciever operating characteristic
# ## auc: area under curve
# ## fpr: false positive rate
# ## tpr: true positive rate
# fpr, tpr, thresholds = roc_curve(train_labels, scores)
# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')

# plot_roc_curve(fpr, tpr, "STRUMPACK")

# auc_score = roc_auc_score(train_labels, scores)
# print("        auc score =", auc_score)



# # stochastic gradient descent classifier from scikit-learn
# from sklearn.linear_model import SGDClassifier
# sgd = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
# sgd.fit(train_points, train_labels)
# pred_sgd = sgd.predict(test_points)
# print('SGD quality =', quality(pred_sgd, test_labels), '%')
# scores = cross_val_predict(
#     sgd, train_points, train_labels, cv=3, method="decision_function")
# auc_score = roc_auc_score(train_labels, scores)
# print("        auc score =", auc_score)
# fpr, tpr, thresholds = roc_curve(train_labels, scores)
# plt.plot(fpr, tpr, "b:", label="SGD")
# plt.legend(loc="lower right")
# plt.show()


