import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import argparse

from classy import Class

import initial
from powerbispectrum import ComputePowerBiSpectrum

parser = argparse.ArgumentParser(description='Computation of power and bispectrum kernels')
parser.add_argument('-cosmo', type=int, help='number of test cosmology', required=True)
parser.add_argument('-estimator', type=str,help='pk or bk', required=True)
parser.add_argument('-spectrum_part', type=str, default='tree',help='tree, SN, PNG for bk or tree, 1loop, counterterm for pk', required=False)
parser.add_argument('-kernel', type=int, help='number of kernel to emulate in list', required=True)
parser.add_argument('-ells', type=str, help='multipoles of power or bispectrum - (ell1 ell2) ELL', required=True)
parser.add_argument('-diag', type=str, default='Yes', help='file with full kernel or diag only for bk', required=False)
parser.add_argument('-directory', type=str, default='/home/rneveux/to_emulator_diag_kcut/with_f/', help='name of directory', required=False)
cmdline = parser.parse_args()

k=np.arange(0.005,0.2025,.0025)

z=.8
if cmdline.estimator == 'bk':
        if cmdline.spectrum_part == 'tree':
                all_kernels =   [
                                'b1_b1_b1', 'b1_b1_b2','b1_b1_bG2','b1_b1_f','b1_b1_b1_f','b1_b1_f_f',
                                'b1_b2_f','b1_bG2_f','b1_f_f',
                                'b1_f_f_f','b2_f_f','bG2_f_f','f_f_f','f_f_f_f',
                                'c1_b1_b1','c1_b1_b2','c1_b1_bG2','c1_b1_f','c1_b1_b1_f','c1_b1_f_f','c1_b2_f',
                                'c1_bG2_f','c1_f_f','c1_f_f_f','c1_c1_b1','c1_c1_b2','c1_c1_bG2','c1_c1_f',
                                'c1_c1_b1_f','c1_c1_f_f','c2_b1_b1','c2_b1_b2','c2_b1_bG2','c2_b1_f','c2_b1_b1_f',
                                'c2_b1_f_f','c2_b2_f','c2_bG2_f','c2_f_f','c2_f_f_f','c2_c1_b1','c2_c1_b2',
                                'c2_c1_bG2','c2_c1_f','c2_c1_b1_f','c2_c1_f_f','c2_c2_b1','c2_c2_b2','c2_c2_bG2',
                                'c2_c2_f','c2_c2_b1_f','c2_c2_f_f'
                                ]
        elif cmdline.spectrum_part == 'SN':
                all_kernels =   [
                                'Bshot_b1_b1', 'Bshot_b1_f', 'Bshot_b1_c1', 'Bshot_b1_c2', 
                                'Pshot_f_b1', 'Pshot_f_f', 'Pshot_f_c1', 'Pshot_f_c2'
                                ]
        elif cmdline.spectrum_part == 'PNG':
                all_kernels =   [
                                'fnlloc_b1_b1_b1', 'fnlloc_b1_b1_f', 'fnlloc_b1_f_f', 'fnlloc_f_f_f', 
                                'fnlequi_b1_b1_b1', 'fnlequi_b1_b1_f', 'fnlequi_b1_f_f', 'fnlequi_f_f_f',
                                'fnlortho_b1_b1_b1', 'fnlortho_b1_b1_f', 'fnlortho_b1_f_f', 'fnlortho_f_f_f'
                                ]
        ell1 = int(cmdline.ells[0])
        ell2 = int(cmdline.ells[1])
        ELL = int(cmdline.ells[2])

elif cmdline.estimator == 'pk':
        if cmdline.spectrum_part == 'tree':
                all_kernels =   [
                                "b1_b1", "b1_f","f_f",
                                ]
        elif cmdline.spectrum_part == '1loop':
                all_kernels =   [
                                "b2_b2","b2_bG2","b1_b2","b2_f","b1_b2_f","b2_f_f","bG2_bG2","b1_bG2","bG2_f",
                                "b1_bG2_f","bG2_f_f","b1_b1","b1_f","b1_b1_f","b1_f_f","f_f","f_f_f","b1_b1_f_f","b1_f_f_f","f_f_f_f",

                                "b1_b3","b1_bG3","b1_bG2d","b1_bGamma3","b1_b1","b1_f","b1_b1_f_f","b1_f_f_f","b1_b1_f",
                                "b1_f_f","b1_b2","b1_bG2","b1_b2_f","b1_bG2_f","b3_f","bG3_f","bG2d_f", "bGamma3_f","f_f",
                                "f_f_f_f","f_f_f","b2_f","bG2_f","b2_f_f","bG2_f_f"
                                ]
        elif cmdline.spectrum_part == 'counterterm':
                all_kernels =   [
                                "c0_b1","c0_f","c1_b1","c1_f","c2_b1","c2_f",
                                ]
        ELL = int(cmdline.ells[0])

cosmo = np.load('/home/rneveux/bispectrum/theory/eft_cosmologies_noDQ1.npy')[cmdline.cosmo]

params_cosmo = {'output': 'tCl mPk','z_max_pk': 3.,'P_k_max_h/Mpc': 50.,
                'omega_cdm':cosmo[0],'omega_b':cosmo[1],'h':cosmo[2],
                'ln10^{10}A_s':cosmo[3],'n_s':cosmo[4]}

c=ComputePowerBiSpectrum(params_cosmo,z)
c.intial_power_spectrum()

name_file = os.path.join(cmdline.directory,'cosmo_{}'.format(str(cmdline.cosmo)),cmdline.ells,cmdline.spectrum_part,all_kernels[cmdline.kernel])
if cmdline.estimator == 'bk':
        c.kernel_computation_Bk(all_kernels[cmdline.kernel],k,ell1,ell2,ELL,integrand=cmdline.spectrum_part,to_save=name_file)
elif cmdline.estimator == 'pk':
        c.kernel_computation_Pk(all_kernels[cmdline.kernel],k,ELL,integrand=cmdline.spectrum_part,to_save=name_file)
