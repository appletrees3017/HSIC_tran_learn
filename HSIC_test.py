from abc import abstractmethod
import time
from scipy.stats import norm as normaldist
from scipy.linalg import sqrtm
import numpy as np
from numpy import fill_diagonal, zeros, shape, mean, sqrt, inv
#permutation导入修改
from numpy.random import permutation

from kernel import Kernel

class TestObject(object):
    def __init__(self, test_type, streaming=False, freeze_data=False):
        self.test_type=test_type
        self.streaming=streaming
        self.freeze_data=freeze_data
        if self.freeze_data:
            self.generate_data()
            assert not self.streaming
    
    @abstractmethod
    def compute_Zscore(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate_data(self):
        raise NotImplementedError
    
    def compute_pvalue(self):
        Z_score = self.compute_Zscore()
        pvalue = normaldist.sf(Z_score)
        return pvalue
    
    def perform_test(self, alpha):
        pvalue=self.compute_pvalue()
        return pvalue<alpha

class HSICTestObject(TestObject):
    def __init__(self, num_samples, data_generator=None, kernelX=None, kernelY=None, kernelZ = None,
                    kernelX_use_median=False,kernelY_use_median=False,kernelZ_use_median=False,
                    rff=False, num_rfx=None, num_rfy=None, induce_set=False, 
                    num_inducing_x = None, num_inducing_y = None,
                    streaming=False, freeze_data=False):
        TestObject.__init__(self,self.__class__.__name__,streaming=streaming, freeze_data=freeze_data)
        self.num_samples = num_samples #We have same number of samples from X and Y in independence testing
        self.data_generator = data_generator
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.kernelX_use_median = kernelX_use_median #indicate if median heuristic for Gaussian Kernel should be used
        self.kernelY_use_median = kernelY_use_median
        self.kernelZ_use_median = kernelZ_use_median
        self.rff = rff
        self.num_rfx = num_rfx
        self.num_rfy = num_rfy
        self.induce_set = induce_set
        self.num_inducing_x = num_inducing_x
        self.num_inducing_y = num_inducing_y
        if self.rff|self.induce_set: 
            self.HSICmethod = self.HSIC_with_shuffles_rff
        else:
            self.HSICmethod = self.HSIC_with_shuffles
    
    def generate_data(self,isConditionalTesting = False):
        if not isConditionalTesting:
            self.data_x, self.data_y = self.data_generator(self.num_samples)
            return self.data_x, self.data_y
        else: 
            self.data_x, self.data_y, self.data_z = self.data_generator(self.num_samples)
            return self.data_x, self.data_y, self.data_z
        ''' for our SimDataGen examples, one argument suffice'''
    
    
    @staticmethod
    def HSIC_U_statistic(Kx,Ky):
        m = shape(Kx)[0]
        fill_diagonal(Kx,0.)
        fill_diagonal(Ky,0.)
        K = np.dot(Kx,Ky)
        first_term = np.trace(K) / (m * (m - 3.0))
        second_term = np.sum(Kx)*np.sum(Ky)/float(m*(m-3.)*(m-1.)*(m-2.))
        third_term = 2.*np.sum(K)/float(m*(m-3.)*(m-2.))
        return first_term+second_term-third_term
    
    
    @staticmethod
    def HSIC_V_statistic(Kx,Ky):
        Kxc=Kernel.center_kernel_matrix(Kx)
        Kyc=Kernel.center_kernel_matrix(Ky)
        return np.sum(Kxc*Kyc)
    
    @staticmethod
    def HSIC_U_statistic_rff(phix,phiy):
        m=shape(phix)[0]
        phix_c=phix-mean(phix,axis=0)
        phiy_c=phiy-mean(phiy,axis=0)
        cov_matrix = (phix_c.T).dot(phiy_c)/float(m-1)
        cov_squared = np.square(cov_matrix**2)
        diag_corelation = np.sum(phix_c**2, axis=0) .dot(np.sum(phiy_c**2, axis=0))/float((m-1)*m)
        return cov_squared-diag_corelation
    
    @staticmethod
    def HSIC_V_statistic_rff(phix,phiy):
        m=shape(phix)[0]
        phix_c=phix-mean(phix,axis=0)
        phiy_c=phiy-mean(phiy,axis=0)
        featCov=(phix_c.T).dot(phiy_c)/float(m)
        return np.linalg.norm(featCov)**2
    
    def HSIC_with_shuffles(self,data_x=None,data_y=None,unbiased=True,num_shuffles=0,
                            estimate_nullvar=False,isBlockHSIC=False):
        start = time.perf_counter()
        if data_x is None:
            data_x=self.data_x
        if data_y is None:
            data_y=self.data_y
        time_passed = time.perf_counter() - start
        if isBlockHSIC:
            Kx, Ky = self.compute_kernel_matrix_on_dataB(data_x,data_y)
        else:
            Kx, Ky = self.compute_kernel_matrix_on_data(data_x,data_y)
        ny=shape(data_y)[0]
        if unbiased:
            test_statistic = HSICTestObject.HSIC_U_statistic(Kx,Ky)
        else:
            test_statistic = HSICTestObject.HSIC_V_statistic(Kx,Ky)
        null_samples=zeros(num_shuffles)
        for jj in range(num_shuffles):
            pp = permutation(ny)
            Kpp = Ky[pp,:][:,pp]
            if unbiased:
                null_samples[jj]=HSICTestObject.HSIC_U_statistic(Kx,Kpp)
            else:
                null_samples[jj]=HSICTestObject.HSIC_V_statistic(Kx,Kpp)
        if estimate_nullvar:
            nullvarx, nullvary = self.unbiased_HSnorm_estimate_of_centred_operator(Kx,Ky)
            nullvarx = 2.* nullvarx
            nullvary = 2.* nullvary
        else:
            nullvarx, nullvary = None, None
        return test_statistic,null_samples,nullvarx,nullvary,Kx, Ky, time_passed
    
    
    
    def HSIC_with_shuffles_rff(self,data_x=None,data_y=None,
                                unbiased=True,num_shuffles=0,estimate_nullvar=False):
        start = time.perf_counter()
        if data_x is None:
            data_x=self.data_x
        if data_y is None:
            data_y=self.data_y
        time_passed = time.perf_counter() - start
        if self.rff:
            phix, phiy = self.compute_rff_on_data(data_x,data_y)
        else:
            phix, phiy = self.compute_induced_kernel_matrix_on_data(data_x,data_y)
        ny=shape(data_y)[0]
        if unbiased:
            test_statistic = HSICTestObject.HSIC_U_statistic_rff(phix,phiy)
        else:
            test_statistic = HSICTestObject.HSIC_V_statistic_rff(phix,phiy)
        null_samples=zeros(num_shuffles)
        for jj in range(num_shuffles):
            pp = permutation(ny)
            if unbiased:
                null_samples[jj]=HSICTestObject.HSIC_U_statistic_rff(phix,phiy[pp])
            else:
                null_samples[jj]=HSICTestObject.HSIC_V_statistic_rff(phix,phiy[pp])
        if estimate_nullvar:
            raise NotImplementedError()
        else:
            nullvarx, nullvary = None, None
        return test_statistic, null_samples, nullvarx, nullvary,phix, phiy, time_passed
    
    
    def get_spectrum_on_data(self, Mx, My):
        '''Mx and My are Kx Ky when rff =False
        Mx and My are phix, phiy when rff =True'''
        if self.rff|self.induce_set:
            Cx = np.cov(Mx.T)
            Cy = np.cov(My.T)
            lambdax=np.linalg.eigvalsh(Cx)
            lambday=np.linalg.eigvalsh(Cy)
        else:
            Kxc = Kernel.center_kernel_matrix(Mx)
            Kyc = Kernel.center_kernel_matrix(My)
            lambdax=np.linalg.eigvalsh(Kxc)
            lambday=np.linalg.eigvalsh(Kyc)
        return lambdax,lambday
    
    
    @abstractmethod
    def compute_kernel_matrix_on_data(self,data_x,data_y):
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        Kx=self.kernelX.kernel(data_x)
        Ky=self.kernelY.kernel(data_y)
        return Kx, Ky
    
    
    @abstractmethod
    def compute_kernel_matrix_on_dataB(self,data_x,data_y):
        Kx=self.kernelX.kernel(data_x)
        Ky=self.kernelY.kernel(data_y)
        return Kx, Ky
    
    @abstractmethod
    def compute_rff_on_data(self,data_x,data_y):
        self.kernelX.rff_generate(self.num_rfx,dim=shape(data_x)[1])
        self.kernelY.rff_generate(self.num_rfy,dim=shape(data_y)[1])
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        phix = self.kernelX.rff_expand(data_x)
        phiy = self.kernelY.rff_expand(data_y)
        return phix, phiy
    
    
    @abstractmethod
    def compute_induced_kernel_matrix_on_data(self,data_x,data_y):
        '''Z follows the same distribution as X; W follows that of Y.
        The current data generating methods we use 
        generate X and Y at the same time. '''
        size_induced_set = max(self.num_inducing_x,self.num_inducing_y)
        #print "size_induce_set", size_induced_set
        if self.data_generator is None:
            subsample_idx = np.random.randint(self.num_samples, size=size_induced_set)
            self.data_z = data_x[subsample_idx,:]
            self.data_w = data_y[subsample_idx,:]
        else:
            self.data_z, self.data_w = self.data_generator(size_induced_set)
            self.data_z[[range(self.num_inducing_x)],:]
            self.data_w[[range(self.num_inducing_y)],:]
        #print 'Induce Set'
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        Kxz = self.kernelX.kernel(data_x,self.data_z)
        Kzz = self.kernelX.kernel(self.data_z)
        #R = inv(sqrtm(Kzz))
        R = inv(sqrtm(Kzz + np.eye(np.shape(Kzz)[0])*10**(-6)))
        phix = Kxz.dot(R)
        Kyw = self.kernelY.kernel(data_y,self.data_w)
        Kww = self.kernelY.kernel(self.data_w)
        #S = inv(sqrtm(Kww))
        S = inv(sqrtm(Kww + np.eye(np.shape(Kww)[0])*10**(-6)))
        phiy = Kyw.dot(S)
        return phix, phiy
    
    
    def compute_pvalue(self,data_x=None,data_y=None):
        pvalue,_=self.compute_pvalue_with_time_tracking(data_x,data_y)

        return pvalue


