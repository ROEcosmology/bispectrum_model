import os
import numpy as np
from matplotlib import pyplot as plt
from classy import Class

import initial
import bispec_kernels

class ComputeKernels():

	def __init__(self, k, params_cosmo, z, ell1=0, ell2=0, ELL=0, diag=True):

		self.params_cosmo = params_cosmo
		self.k = k
		self.z = z
		self.ell1 = ell1
		self.ell2 = ell2
		self.ELL = ELL
		self.diag = diag


	def intial_power_spectrum(self):

		cosmo = Class()
		cosmo.set(self.params_cosmo)
		cosmo.compute()

		self.initial_cosmo = initial.InputPowerSpectrum(self.z, cosmo, params_fid=self.params_cosmo)
		self.initial_cosmo.calcMatterPowerSpectrum()
		self.k_in, self.pk_in = self.initial_cosmo.getMatterPowerSpectrum()

		self.f = self.initial_cosmo.getGrowthRate()
		self.rs_drag = cosmo.rs_drag()

	def kernel_computation(self,kernel_name,ell1=0,ell2=0,ELL=0,to_save=False):

		if self.diag: kernel = bispec_kernels.ClassBiSpectrumKernelDiag(self.k_in,self.pk_in,self.f,self.rs_drag)
		else: kernel = bispec_kernels.ClassBiSpectrumKernel(self.k_in,self.pk_in,self.f,self.rs_drag)
		kernel.set_normalization(self.initial_cosmo.getSigma8ForNormalization())

		self.K = kernel.calc_K(name=kernel_name,
		kbin=self.k,ell1=ell1,ell2=ell2,ELL=ELL)

		if to_save: 
			self.K['params_cosmo']=self.params_cosmo
			self.K['z']=self.z
			self.save(to_save)

	def save(self,to_save):

		os.makedirs(os.path.dirname(to_save), exist_ok=True)
		np.save(to_save,self.K)

	def load_kernels(self,directory,kernels):

		self.kernels={}
		for b in kernels:
			f = os.path.join(directory,b+'.npy')
			tmp = np.load(f,allow_pickle=True).item()
			self.kernels[b] = tmp['K']

	def bispectrum_model(self,**kwargs):
		'''pooling of the different kernels associated with given bias parameters'''

		for b in kwargs: setattr(self,b,kwargs[b])

		if self.diag: self.bispectrum = np.zeros((len(self.k)))
		else: self.bispectrum = np.zeros((len(self.k),len(self.k)))

		for b in self.kernels:

			bias = 1
			if 'b1' in b: bias *= self.b1**b.count('b1')
			if 'b2' in b: bias *= self.b2
			if 'bG2' in b: bias *= self.bG2
			if 'f' in b: bias *= self.f**b.count('f')
			if 'c1' in b: bias *= self.c1**b.count('c1')
			if 'c2' in b: bias *= self.c2**b.count('c2')

			self.bispectrum += self.kernels[b]*bias

		return self.bispectrum

	def plot_bispectrum(self,plot_save,k2='diag',f='pdf'):

		if k2=='diag': B = np.diag(self.bispectrum)
		else: B = self.bispectrum[:,k2]
		plt.plot(self.k,self.k**2*B)
		plt.xlabel(r'$k~[h.\mathrm{Mpc}^{-1}]$',fontsize=16)
		plt.ylabel(r'$k^2B_{%(l1)s%(l2)s%(l)s}~[(h^{-1}.Mpc)^{4}]$'%{'l1':self.ell1,'l2':self.ell2,'l':self.ELL},fontsize=16)
		plt.savefig(plot_save, format=f)


