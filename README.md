个人代码学习--[非传播]
代码参考自--https://github.com/oxcsml/kerpy/tree/master/independence_testing  [尝试复现]
============================修改了一些不兼容部分+++++++++++++++++++++++++++++
1.在较新的numpy版本中，permutation函数位于numpy.random模块中，而不是直接从numpy导入。---只调整permutation的导入：from numpy.random import permutation
2.time.clock() 已在 Python 3.3 弃用，Python 3.8 中移除 ---start = time.perf_counter() \n elapsed = time.perf_counter() - start （使用高精度计时器）
3.// 运算符在整数除法时返回整数，但部分浮点运算需要显式转换 ---first_term = np.trace(K) / (m * (m - 3.0))
4.问题​​：from numpy import mean, sum, zeros 不符合 PEP8 规范
5..HSIC_U_Statistic_rff(phix,phiy)接口补充  
 ..HSICTestObject 类中 @abstractmethod compute_kernel_matrix_on_data 不应该有具体实现方法--去掉@abstractmethod
6.numpy.random.permutation 应使用随机状态对象  rng = np.random.default_rng() \n X = X[rng.permutation(n)[:1000], :] (GaussianKernel get_sigma_median_heuristic)
