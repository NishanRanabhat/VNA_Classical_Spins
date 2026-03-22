import numpy as np 
from data_set import ScalingDataset
from dynamical_fss import FiniteSizeScaling
from utilities import gather_data
from compute_errors import parametric_bootstrap, compute_systematic_error

"""
This is a mock example to use the code
"""

#prepare data
#each dataset should have a system size L, the domain x, the range y, and the error in y
#then use these to create a ScalingDataset object

N = [16, 32, 64, 128, 256]
taus = [48, 64, 128, 192, 256, 384, 512, 640, 768, 896, 1024, 1536, 2048, 3072,
        4096, 5120, 6144, 7168, 8192, 12288, 16384, 18432, 25251, 32768]

sample_size = 32768
layer = 1
Nh = 16
T = 1.0

N1 = N[0]
x1 = np.array(taus)
y1, var1 = gather_data(sample_size, layer, Nh, T, N1, taus, mode="VQAtrain",
                       interaction="fully_connected", observable="mag2")
err1 = np.sqrt(var1)/np.sqrt(float(sample_size))
ds1 = ScalingDataset(N1,x1,y1,err1) 

N2 = N[1]
x2 = np.array(taus)
y2, var2 = gather_data(sample_size, layer, Nh, T, N2, taus, mode="VQAtrain",
                       interaction="fully_connected", observable="mag2")
err2 = np.sqrt(var2)/np.sqrt(float(sample_size))
ds2 = ScalingDataset(N2,x2,y2,err2) 

N3 = N[2]
x3 = np.array(taus)
y3, var3 = gather_data(sample_size, layer, Nh, T, N3, taus, mode="VQAtrain",
                       interaction="fully_connected", observable="mag2")
err3 = np.sqrt(var3)/np.sqrt(float(sample_size))
ds3 = ScalingDataset(N3,x3,y3,err3)

N4 = N[3]
x4 = np.array(taus)
y4, var4 = gather_data(sample_size, layer, Nh, T, N4, taus, mode="VQAtrain",
                       interaction="fully_connected", observable="mag2")
err4 = np.sqrt(var4)/np.sqrt(float(sample_size))
ds4 = ScalingDataset(N4,x4,y4,err4)

N5 = N[4]
x5 = np.array(taus)
y5, var5 = gather_data(sample_size, layer, Nh, T, N5, taus, mode="VQAtrain",
                       interaction="fully_connected", observable="mag2")
err5 = np.sqrt(var5)/np.sqrt(float(sample_size))
ds5 = ScalingDataset(N5,x5,y5,err5)

#initiate a FiniteSizeScaling object

s_factor = 1.0 #spline smoothing factor multiplier (default 1.0)
k = 3 #spline degree (default 3)
method = 'Nelder-Mead' #optimizer method for fitting (default 'Nelder-Mead')
maxiter = 1000 #maximum iterations for optimizer (default 1000)
a0, b0 = 1.0, 0.5 #initial guesses for scaling exponents a and b

fss = FiniteSizeScaling(ds1, ds2, ds3, ds4, ds5,
                        s_factor=s_factor,k=k,method=method,maxiter=maxiter,a0=a0,b0=b0)

#call fit to extract the critical exponents a and b corresponding to the best collapse
a,b = fss.fit()
print(f"Fitted critical exponents: a = {a}, b = {b}")

parametric_bootstrap(Ls=N, x_list=[np.array(taus)] * len(N), 
                     y_det_list = [y1, y2, y3, y4, y5],
                    err_list = [err1, err2, err3, err4, err5], 
                    a0=a0, b0=b0, s_factor=s_factor, k=k)

compute_systematic_error(N, [np.array(taus)] * len(N), [y1, y2, y3, y4, y5], [err1, err2, err3, err4, err5], a0, b0, s_factor, k)

