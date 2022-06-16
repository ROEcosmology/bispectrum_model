import numpy as np
from matplotlib import pyplot as plt
from powerbispectrum import ComputePowerBiSpectrum

k=np.logspace(-3,0,100)
params_cosmo = {# fixed LambdaCDM parameters
                   'A_s':2.089e-9,
                   'n_s':0.9649,
                   'tau_reio':0.052,
                   'omega_b':0.02237,
                   'omega_cdm':0.12,
                   'h':0.6736,
                   'YHe':0.2425,
#                     'N_eff':3.046,
                    'N_ur':2.0328,
                    'N_ncdm':1,
                    'm_ncdm':0.06,
                   # other output and precision parameters
                    'output': 'tCl mPk','z_max_pk': 3.,
                    'P_k_max_h/Mpc': 50.,}
c=ComputePowerBiSpectrum(params_cosmo,.61)
c.intial_power_spectrum()

pk = np.zeros(len(k))
c.calc_P(
            k, 0,
            alpha_perp=1, alpha_parallel=1, b1=2, b2=0, bG2=0, b3=0, bG3=0, bG2d=0, bGamma3=0,
            c0=0, c1=0, c2=0, knl=.3,
            integrand='tree',
            ks=.05
        )
plt.plot(c.PK['kbin'],c.PK['K'],label='tree')
pk += c.PK['K']
c.calc_P(
            k, 0,
            alpha_perp=1, alpha_parallel=1, b1=2, b2=0, bG2=0, b3=0, bG3=0, bG2d=0, bGamma3=0,
            c0=1, c1=0, c2=0, knl=.3,
            integrand='counterterm',
            ks=.05
        )
plt.plot(c.PK['kbin'],c.PK['K'],label='counterterm')
pk += c.PK['K']
c.calc_P(
            k, 0,
            alpha_perp=1, alpha_parallel=1, b1=2, b2=-1, bG2=0.1, b3=0, bG3=0, bG2d=0, bGamma3=-0.1,
            c0=0, c1=0, c2=0, knl=.3,
            integrand='1loop',
            ks=.05
        )
plt.plot(c.PK['kbin'],c.PK['K'],label='1loop')
plt.plot(c.PK['kbin'],-c.PK['K'],label='1loop-neg')
pk += c.PK['K']
plt.plot(c.PK['kbin'],pk,label='sum')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.show()
