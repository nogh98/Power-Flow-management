# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
def calculate_reward(Pbss, Pg, Eb2, Eb1, Tarif, Capacity = 100, cbt = 400):

    """

    Computing the reward R = g(xk, ak, xk+1)
    R = 1/(Cbd + Cg)
    Cbd = Cbt * max {Cbd,T , Cbd,DoD , Cbd,SoC}
    Cg = Tarif * Pg
    Cbd,T = (1/800) * abs(Pbss) + 1/50
    Cbd,Dod = abs(1/L(DoD2) - 1/L(DoD1)) 
    L(DoD) = 694 * (DoD) ** 0.795
    DoD = 1 - SoC
    SoC = Eb / Capacity
    Cbd, SoC = m * SoC(avg) - d / (Qfade * n * Yh)
    m = 1.59e-5, d = 6.41e-6, Qfade = 0.2, n = 15, Yh = 8760


    """
    m = 1.59e-5
    d = 6.41e-6
    Qfade = 0.2

    n = 15
    Yh = 8760
    soc1 = Eb1 / Capacity
    soc2 = Eb2 / Capacity

    dod1 = 1 - soc1
    dod2 = 1 - soc2

    L = lambda dod: 694 * (dod) ** -0.795
    
    soc_avg = (soc1 + soc2) / 2
    
    Cbd_soc = -0.005 * (soc_avg) ** 2 + 0.0542 * soc_avg - 0.0019 #(m * soc_avg - d) / (Qfade * n * Yh)
    
    Cbd_dod = np.abs((1 / L(dod1)) - (1 / L(dod2)))
    
    Cbd_T = 0.02 + np.abs(Pbss) * (1 / 800)
    
    
    
    Cbd_total = max(Cbd_soc, Cbd_dod, Cbd_T)
    
    Cbd = cbt * Cbd_total
    
    Cg = Tarif * Pg
    
    #print('Cbd_soc is: %.5f | Cbd_dod is: %.5f | Cbd_T is: %.5f | Cg is: %.5f.'%(Cbd_soc, Cbd_dod, Cbd_T, Cg))
    
    #print('The value of reward is: %.5f'%(1/(Cbd + Cg + 1)))
    return 1/(Cbd + Cg + 1)