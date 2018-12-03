import numpy as np
import ctypes
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
sp = ctypes.cdll.LoadLibrary(
    '/home/pieterg/LBL/STRUMPACK/STRUMPACK/'
    'install/lib/libstrumpack.so')


class STRUMPACKKernel(BaseEstimator, ClassifierMixin):

    # use same names/types of kernels as in the SVC code?
    def __init__(self, h=1., lam=4., kernel='rbf',
                 approximation='HSS', mpi=False, argv=None):
        self.h = h
        self.lam = lam
        self.kernel = kernel
        self.approximation = approximation
        self.mpi = mpi
        self.argv = argv


    def __del__(self):
        try: sp.STRUMPACK_destroy_kernel_double(self.K_)
        except: pass


    def fit(self, X, y):
        # TODO make sure that X and y are float64 aka double
        if X.dtype is not np.dtype('double'):
            print("ERROR: STRUMPACKKernel expects "
                  "double precision floating point")

        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # store the classes seen during fit
        self.classes_ = unique_labels(y)

        k = 0
        if self.kernel == 'rbf' or self.kernel == 'Gauss': k = 0
        elif self.kernel == 'Laplace': k = 1
        else:
            print("Warning: Kernel type", self.kernel, "not recognized")
            print("         Possible values are 'rbf'/'Gauss' or 'Laplace'")
            print("         Using the default 'rbf' kernel")
        if self.approximation is 'HODLR' and self.mpi is False:
            print("ERROR: HODLR requires mpi=True")
        self.K_ = sp.STRUMPACK_create_kernel_double(
            ctypes.c_int(X.shape[0]), ctypes.c_int(X.shape[1]),
            ctypes.c_void_p(X.ctypes.data),
            ctypes.c_double(self.h), ctypes.c_double(self.lam),
            ctypes.c_int(k))
        if self.argv==None: self.argv=[]
        LP_c_char = ctypes.POINTER(ctypes.c_char)
        argc = len(self.argv)
        argv = (LP_c_char * (argc + 1))()
        for i, arg in enumerate(self.argv):
            enc_arg = arg.encode('utf-8')
            argv[i] = ctypes.create_string_buffer(enc_arg)
        if self.approximation == 'HSS':
            if self.mpi:
                sp.STRUMPACK_kernel_fit_HSS_MPI_double(
                    self.K_, ctypes.c_void_p(y.ctypes.data),
                    ctypes.c_int(argc), argv)
            else:
                sp.STRUMPACK_kernel_fit_HSS_double(
                    self.K_, ctypes.c_void_p(y.ctypes.data),
                    ctypes.c_int(argc), argv)
        elif self.approximation == 'HODLR':
            sp.STRUMPACK_kernel_fit_HODLR_MPI_double(
                self.K_, ctypes.c_void_p(y.ctypes.data),
                ctypes.c_int(argc), argv)
        else:
            print("Warning: Approximation type", self.approximation,
                  "not recognized")
            print("         Possible values are 'HSS' or 'HODLR'")
            print("         Using the default 'HSS' kernel")
            sp.STRUMPACK_kernel_fit_HSS_double(
                self.K_, ctypes.c_void_p(y.astype(np.float64).ctypes.data),
                ctypes.c_int(argc), argv)
        # return the classifier
        return self


    def predict(self, X):
        # TODO make sure there are only 2 classes?
        check_is_fitted(self, 'K_')
        prediction = np.zeros((X.shape[0],1))
        sp.STRUMPACK_kernel_predict_double(
            self.K_, ctypes.c_int(X.shape[0]),
            ctypes.c_void_p(X.ctypes.data),
            ctypes.c_void_p(prediction.ctypes.data))
        return [self.classes_[0] if prediction[i] < 0.0 else self.classes_[1]
                for i in range(X.shape[0])]

    def decision_function(self, X):
        check_is_fitted(self, 'K_')
        prediction = np.zeros((X.shape[0],1))
        sp.STRUMPACK_kernel_predict_double(
            self.K_, ctypes.c_int(X.shape[0]),
            ctypes.c_void_p(X.ctypes.data),
            ctypes.c_void_p(prediction.ctypes.data))
        return prediction
