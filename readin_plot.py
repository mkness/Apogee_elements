#!/usr/bin/python 
import scipy 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle 
import matplotlib.cm as cm 
import pyfits 
import glob 
from glob import glob 
import numpy 
import pickle
from numpy import * 
from scipy import ndimage
from scipy import interpolate 
from numpy import loadtxt
import os 
import numpy as np
from numpy import * 
import matplotlib 
from pylab import rcParams
from pylab import * 
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.pyplot import axes
from matplotlib.pyplot import colorbar
#from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
s = matplotlib.font_manager.FontProperties()
s.set_family('serif')
s.set_size(14)
from matplotlib import rc
rc('text', usetex=False)
rc('font', family='serif')
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams['figure.figsize'] = 14.0, 8.0
f, ((ax1, ax2, ax3,ax4,ax5, ax6, ax7, ax8),(ax9,ax10,ax11,ax12,ax13,ax14,ax15, ax16), (ax17,ax18,ax19,ax20, ax21, ax22, ax23, ax24)) = plt.subplots(ncols=8, nrows =3)

labels = ["teff", "logg", "feh", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Ti", "V", "Mn", "Ni", "P", "Cr", "Co", "Cu", "Rb", "blank", "blank"]
def returnscatter(diffxy):
    rms = (np.sum([ (val)**2  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms
axes =  [ax1,ax2,ax3,ax4,ax5,ax6, ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15, ax16, ax17, ax18, ax19, ax20, ax21,ax22, ax23, ax24]

for ax,b,c2,lab in zip(axes, inputs, outputs_rescaled, labels):
  #ax.plot(b, c, 'ko',ms = 3, alpha = 0.4)
  ax.scatter(b, c2, c = list(inputs[0]), vmin= 3600, vmax = 5700, linewidth =0, s = 12) 
  minval = min(b)*0.8
  maxval = max(b)*1.2
  ax.set_xlim(minval, maxval)
  ax.set_ylim(minval, maxval)
  ax.plot([minval, maxval], [minval, maxval], 'k') 
  bs,scat = returnscatter(b-c2) 
  scat = str(round(scat, 2) )
  bias = str(round(bs, 3) )
  ax.text(0.3,0.8, '$\sigma$='+scat, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes, fontsize=8) 
  ax.text(0.3,0.9, "bias="+bias, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes, fontsize=8) 
  ax.set_title(lab) 
  ax.set_title(lab) 

f.subplots_adjust(hspace=0.25)
f.subplots_adjust(bottom=0.10)
#show()
#draw()

#savefig('dr13_selftest.pdf', bbox = 'tight', fmt = "pdf") 
