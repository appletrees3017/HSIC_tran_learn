from abc import abstractmethod
from numpy import eye, shape, dot, cos, sin, sqrt
from numpy import reshape, median
from numpy import concatenate, zeros, ones, fill_diagonal
from numpy.random import permutation,randn
from numpy.linalg import norm   # for norm function

from scipy.spatial.distance import squareform, pdist, cdist

import numpy as np
import warnings


def check_type(varvalue, varname, vartype, required_shapelen=None):
    if not type(varvalue) is vartype:
        raise TypeError("Variable " + varname + " must be of type " + vartype.__name__ + \
                            ". Given is " + str(type(varvalue)))
    if not required_shapelen is None:
        if not len(varvalue.shape) is required_shapelen:
            raise ValueError("Variable " + varname + " must be " + str(required_shapelen) + "-dimensional")
    return 0

class Kernel(object):
    def __init__(self):
        self.rff_num=None
        self.rff_freq=None
        pass
    
    def __str__(self):
        s=""
        return s
    
    @abstractmethod
    def kernel(self, X, Y=None):
        raise NotImplementedError()
    
    @abstractmethod
    def set_kerpar(self,kerpar):
        self.set_width(kerpar)
    
    @abstractmethod
    def set_width(self, width):
        if hasattr(self, 'width'):
            warnmsg="\nChanging kernel width from "+str(self.width)+" to "+str(width)
            #warnings.warn(warnmsg) ---need to add verbose argument to show these warning messages
            if self.rff_freq is not None:
                warnmsg="\nrff frequencies found. rescaling to width " +str(width)
                #warnings.warn(warnmsg)
                self.rff_freq=self.unit_rff_freq/width
            self.width=width
        else:
            raise ValueError("Senseless: kernel has no 'width' attribute!")
    
    @abstractmethod
    def rff_generate(self,m,dim=1):
        raise NotImplementedError()
    
    @abstractmethod
    def rff_expand(self,X):
        if self.rff_freq is None:
            raise ValueError("rff_freq has not been set. use rff_generate first")
        """
        Computes the random Fourier features for the input dataset X
        for a set of frequencies in rff_freq.
        This set of frequencies has to be precomputed
        X - 2d numpy.ndarray, first set of samples:
            number of rows: number of samples
            number of columns: dimensionality
        """
        check_type(X, 'X',np.ndarray)
        xdotw=dot(X,(self.rff_freq).T)
        return sqrt(2./self.rff_num)*np.concatenate( ( cos(xdotw),sin(xdotw) ) , axis=1 )
        
    @abstractmethod
    def gradient(self, x, Y):
        
        # ensure this in every implementation
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        raise NotImplementedError()
    
    @staticmethod
    def centering_matrix(n):
        """
        Returns the centering matrix eye(n) - 1.0 / n
        """
        return eye(n) - 1.0 / n
    
    @staticmethod
    def center_kernel_matrix(K):
        """
        Centers the kernel matrix via a centering matrix H=I-1/n and returns HKH
        """
        n = shape(K)[0]
        H = eye(n) - 1.0 / n
        return  1.0 / n * H.dot(K.dot(H))
class GaussianKernel(Kernel):
    def __init__(self, sigma=1.0, is_sparse = False):
        Kernel.__init__(self)
        self.width = sigma
        self.is_sparse = is_sparse
    
    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "width="+ str(self.width)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - 2d numpy.ndarray, first set of samples:
            number of rows: number of samples
            number of columns: dimensionality
        Y - 2d numpy.ndarray, second set of samples, can be None in which case its replaced by X
        """
        if self.is_sparse:
            X = X.todense()
            Y = Y.todense()
        check_type(X, 'X',np.ndarray)
        assert(len(shape(X))==2)
        
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            check_type(Y, 'Y',np.ndarray)
            assert(len(shape(Y))==2)
            assert(shape(X)[1]==shape(Y)[1])
            sq_dists = cdist(X, Y, 'sqeuclidean')
        
        K = np.exp(-0.5 * (sq_dists) / self.width ** 2)
        return K
    
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the Gaussian kernel wrt. to the left argument, i.e.
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2), which is
        labla_x k(x,y)=1.0/sigma**2 k(x,y)(y-x)
        Given a set of row vectors Y, this computes the
        gradient for every pair (x,y) for y in Y.
        """
        if self.is_sparse:
            x = x.todense()
            Y = Y.todense()
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        x_2d=reshape(x, (1, len(x)))
        k = self.kernel(x_2d, Y)
        differences = Y - x
        G = (1.0 / self.width ** 2) * (k.T * differences)
        return G
    
    
    def rff_generate(self,m,dim=1):
        self.rff_num=m
        self.unit_rff_freq=randn(int(m/2),dim)
        self.rff_freq=self.unit_rff_freq/self.width
    
    @staticmethod
    def get_sigma_median_heuristic(X, is_sparse = False):
        if is_sparse:
            X = X.todense()
        n=shape(X)[0]
        if n>1000:
            X=X[permutation(n)[:1000],:]
        dists=squareform(pdist(X, 'euclidean'))
        median_dist=median(dists[dists>0])
        sigma=median_dist/sqrt(2.)
        return sigma
    