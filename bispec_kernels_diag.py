# -*- coding: utf-8 -*-
import pyximport

import numpy as np
from scipy import interpolate
import hitomipy
import pycuba
import os


class ClassBiSpectrumKernelDiag(ClassBiSpectrumKernel):

    def Integrand_K(self, ndim, xx, ncomp, ff, userdata):

        self.xx_in = np.zeros(ndim[0])
        self.ff_out = np.zeros(ncomp[0])


        for i in range(ndim[0]):
            self.xx_in[i] = xx[i]

        self.ndim = ndim[0]
        self.ncomp = ncomp[0]

        if self.integrand=='tree':
            hitomipy.integrand_B_Kernel_diag_py(
                    self.xx_in,
                    self.ndim,
                    self.ff_out,
                    self.ncomp,
                    self.kbin,
                    len(self.kbin),
                    self.ell1,
                    self.ell2,
                    self.ELL,
                    self.f,
                    self.Sigma2, 
                    self.dSigma2, 
                    self.name
                )

        elif self.integrand=='SN':
            hitomipy.integrand_B_Kernel_SN_diag_py(
                self.xx_in,
                self.ndim,
                self.ff_out,
                self.ncomp,
                self.kbin,
                len(self.kbin),
                self.ell1,
                self.ell2,
                self.ELL,
                self.f,
                self.Sigma2, 
                self.dSigma2, 
                self.name
            )

        if self.integrand=='PNG':
            hitomipy.integrand_B_Kernel_PNG_diag_py(
                    self.xx_in,
                    self.ndim,
                    self.ff_out,
                    self.ncomp,
                    self.kbin,
                    len(self.kbin),
                    self.ell1,
                    self.ell2,
                    self.ELL,
                    self.f,
                    self.Sigma2, 
                    self.dSigma2, 
                    self.name
                )

        for i in range(ncomp[0]):
                ff[i] = self.ff_out[i]

        return 0

    def calc_K(
        self,
        integrand='tree',
        name,
        kbin=np.linspace(0.01, 0.2, 20),
        ell1=0,
        ell2=0,
        ELL=0,
        ks=.05, #flexible - no impact for physical scale
    ):

        self.integrand = integrand
        (kbin2_out, kbin1_out) = np.meshgrid(kbin, kbin)
        ## flags ##
        output_dict_ini = {
            "kbin1": kbin1_out,
            "kbin2": kbin2_out,
            "K": np.zeros((len(kbin), len(kbin))),
            "ell1": ell1,
            "ell2": ell2,
            "ELL": ELL,
        }

        ## Kernel to compute ##
        self.name = name
       
        ## set kbin ##
        self.kbin = kbin

        ## set multipole indices ##
        self.ell1 = ell1
        self.ell2 = ell2
        self.ELL = ELL

        ## initialization ##
        hitomipy.initializeInputPowerSpectrum_py()

        ## calc. wigner 3j symbols ##
        hitomipy.setWigner3j_py()

        ## read linear power spectrum ##
        hitomipy.readInputPowerSpectrum_py(self.k_temp, self.P_temp, len(self.k_temp))

        ## normalization ##
        hitomipy.calcNormalizationUsingSigma8_py(self.sigma8_norm)
        hitomipy.calcNormalizationNoWiggle_py(1.0)

        # IR resummation damping term
        self.Sigma2 = hitomipy.Sig2_py(self.rs_drag,ks)
        self.dSigma2 = hitomipy.dSig2_py(self.rs_drag,ks)
        
        ## compute bispectra ##
        NDIM = 3
        NCOMP = len(self.kbin)
        if NCOMP > 1024:
            print("# of NCOMP should be <= 1024, otherwise results become zero.")
            return output_dict_ini

        AA = pycuba.Cuhre(
                    self.Integrand_K, NDIM, ncomp=NCOMP, key=0, verbose=0 | 4
                )["results"]

        bk_temp = np.zeros((NCOMP))
        for i in range(NCOMP):
            bk_temp[i] = AA[i]["integral"]

        ## finalize parameters ##
        hitomipy.finalizeInputPowerSpectrum_py()

        #########################

        ## sigma8 normalisation ##
        if self.integrand == 'tree': bk_out = bk_temp*self.sigma8_norm**4
        elif self.integrand == 'SN': bk_out = bk_temp*self.sigma8_norm**2
        elif self.integrand == 'PNG': bk_out = bk_temp*self.sigma8_norm**3

        ## output dict ##
        output_dict = {
            "kbin1": self.kbin,
            "K": bk_out,
            "ell1": ell1,
            "ell2": ell2,
            "ELL": ELL,
            "kernel": self.name
        }

        return output_dict
