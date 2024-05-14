# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 18:53:08 2024

@author: Marrol Faith Ramirez
"""

import scipy.stats as sts
import matplotlib.pyplot as plt

def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale = 0.1)
    return likelihood_out/likelihood_out.sum()

likelihood_out = likelihood_func(1.7, mu)

plt.plot(mu, likelihood_out)
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.show()