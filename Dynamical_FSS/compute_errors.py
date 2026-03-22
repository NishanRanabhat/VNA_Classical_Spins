import numpy as np
from data_set import ScalingDataset
from dynamical_fss import FiniteSizeScaling
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sys import path

def parametric_bootstrap(Ls, x_list, y_det_list, err_list, a0, b0, s_factor, k, 
                        method="Nelder-Mead", maxiter=1000, seed=205, Nboot=500):
    rng = np.random.default_rng(seed)
    boot_ab = []
    trials = 0
    while len(boot_ab) < Nboot:
        trials += 1
        y_syn = [y_det + rng.normal(0, err, size=len(y_det))
                 for y_det, err in zip(y_det_list, err_list)]
        has_neg = any((block < 0).any() for block in y_syn)
        if has_neg:
            continue
        datasets = [ScalingDataset(L, x, y, err)
                    for L, x, y, err in zip(Ls, x_list, y_syn, err_list)]
        fss = FiniteSizeScaling(*datasets, 
                                s_factor=s_factor,
                                k=k,
                                method=method,
                                maxiter=maxiter,
                                a0=a0,
                                b0=b0)
        a_, b_ = fss.fit()
        boot_ab.append((a_, b_))
    boot_ab = np.array(boot_ab)       # shape (Nboot, 2)  
    a_mean, b_mean = boot_ab.mean(axis=0) 
    sigma_a_stat, sigma_b_stat = boot_ab.std(axis=0, ddof=1)
    print("\nresults over", Nboot, "accepted replicas")
    print(f"ā = {a_mean:.6f}   σ_a(stat) = {sigma_a_stat:.6f}")
    print(f"b̄ = {b_mean:.6f}   σ_b(stat) = {sigma_b_stat:.6f}")
    print(f'Bootstrapping completed in {trials} trials.')
    return a_mean, b_mean, sigma_a_stat, sigma_b_stat

def compute_systematic_error(N, taus, y_list, err_list, a0, b0, s_factor, k,
                            method="Nelder-Mead", maxiter=1000):
    ab = np.empty((len(N), 2))
    for i, N_ in enumerate(N):
        N_cut = N.copy()
        y_list_cut = y_list.copy()
        err_list_cut = err_list.copy()
        # Remove one system size
        N_cut.pop(i)
        y_list_cut.pop(i)
        err_list_cut.pop(i)
        datasets = [ScalingDataset(L, tau, y, err)
                    for L, tau, y, err in zip(N_cut, taus, y_list_cut, err_list_cut)]
        fss = FiniteSizeScaling(*datasets, 
                                s_factor=s_factor,
                                k=k,
                                method=method,
                                maxiter=maxiter,
                                a0=a0,
                                b0=b0)
        a_i, b_i = fss.fit()
        ab[i, :] = [a_i, b_i]
        print(f'Leave-one-out fit {i+1}/{len(N)} done (removed N={N_}),' 
              f'a={a_i:.6f}, b={b_i:.6f}')
    a_mean, b_mean = ab.mean(axis=0)
    sigma_a, sigma_b = ab.std(axis=0, ddof=1)
    print("\nresults over systematic error analysis (leave-one-out)")
    print(f"ā = {a_mean:.6f}   σ_a(stat) = {sigma_a:.6f}")
    print(f"b̄ = {b_mean:.6f}   σ_b(stat) = {sigma_b:.6f}")
    print(f'Leave-one-out analysis completed over {len(N)} fits.')
    return a_mean, b_mean, sigma_a, sigma_b

if __name__ == "__main__":
    num_sample=32768
    Tc=1.0 
    layer=1 
    Nh=10
    observable='squared_magnetization'
    Ls = [32, 64, 128, 256]
    x = np.array([4, 5, 7, 8, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 128,
                  196, 256, 280, 324, 364, 400, 448, 512, 1024, 2000, 3000, 
                  4000, 6000, 7000, 8000, 9000, 10000, 16000, 20000, 32000])
    y_det_list = []
    err_list = []
    for L in Ls:
        y, err = load_data(num_sample, L, Tc, layer, Nh, x, observable)
        y_det_list.append(y)
        err_list.append(err)
    parametric_bootstrap(Ls, x, y_det_list, err_list, 
                        a0 = 1.299355597385857, b0= 0.4848605962789953, 
                        s_factor= 2.9101000000019206, k=5, seed=42, Nboot=1000)