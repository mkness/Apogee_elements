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
f, ((ax1, ax2, ax3,ax4,ax5),(ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(ncols=5, nrows =3)

def returnscatter(diffxy):
    rms = (np.sum([ (val)**2  for val in diffxy])/len(diffxy))**0.5
    bias = (np.mean([ (val)  for val in diffxy]))
    return bias, rms

#snr_range = arange(-5,300,10) 
snr_range = [0,5,10,15,20,25,30,40,50,60,70,80,90,100,120,140,160] 
snr_range = [0,5,10,20,30,40,60,80,100,120,140,160,180,200, 220, 240,260,280] 
snr_range = [0,5,10,20,30,40,60,80,100,120,140,160,180,200, 220, 240,260,280] 

bias_t, rms_t, bias_g, rms_g, bias_f, rms_f, bias_a, rms_a = [],[],[],[],[],[],[],[]
bias_tf, rms_tf, bias_gf, rms_gf, bias_ff, rms_ff, bias_af, rms_af = [],[],[],[],[],[],[],[]
bias_t_apogee, rms_t_apogee, bias_g_apogee, rms_g_apogee, bias_f_apogee, rms_f_apogee, bias_a_apogee, rms_a_apogee = [],[],[],[],[],[],[],[]
bias_c_apogee, bias_n_apogee = [],[]
bias_c, bias_n = [],[]
bias_al, rms_al = [],[]
bias_al, rms_alf = [],[]
bias_o, bias_na, bias_s, bias_na, bias_mg, bias_mn, bias_v = [],[], [],[],[],[],[]
rms_o, rms_na, rms_s, rms_na, rms_mg, rms_mn, rms_v = [],[], [],[],[],[],[]
rms_of, rms_naf, rms_sf, rms_naf, rms_mgf, rms_mnf, rms_vf = [],[], [],[],[],[],[]
bias_ni, rms_ni =[],[]
bias_nif, rms_nif =[],[]
snrmean = []
rms_c, rms_n = [],[]
rms_cf, rms_nf = [],[]
rms_n_apogee  = [] 
rms_c_apogee  = [] 
rms_o_apogee  = [] 
bias_o_apogee  = [] 
bias_na_apogee  = [] 
rms_na_apogee  = [] 
rms_al_apogee  = [] 
bias_al_apogee  = [] 
bias_mg_apogee  = [] 
bias_ni_apogee, rms_ni_apogee = [],[]
rms_mg_apogee  = [] 
bias_mg_apogee  = [] 
rms_mn_apogee  = [] 
bias_mn_apogee  = [] 
rms_v_apogee  = [] 
bias_v_apogee  = [] 
rms_s_apogee  = [] 
bias_s_apogee  = [] 
#for a,b,c,d,e in zip(t_diff2,g_diff2,feh_diff2,alpha_diff2):
for i in range(0,len(snr_range)-1):
    snrmin = snr_range[i]
    snrmax = snr_range[i+1]
    take1 = logical_and(snr_val2 >= snrmin, snr_val2 <= snrmax)
    take2 = logical_and(abs(t_diff2_apogee) < 1000, abs(g_diff2_apogee) < 10) 
    take3 = abs(feh_diff2_apogee) < 90.
    take4 = abs(n_diff2_apogee) < 90.
    take5 = abs(c_diff2_apogee) < 90.
    take6 = logical_and(take1, logical_and(take2,take3)) 
    take7 = logical_and(take5, take4)
    take8 = fehallap > -4
    take = logical_and(logical_and(take8, take7), take6) 
    snrmean.append(mean(snr_val2[take]) )
    print len(take[take])
    biast, rmst = returnscatter(t_diff2[take])
    biastf, rmstf = returnscatter(t_diff2f[take])
    biasg,rmsg = returnscatter(g_diff2[take])
    biasgf,rmsgf = returnscatter(g_diff2f[take])
    biasf,rmsf = returnscatter(feh_diff2[take])
    biasff,rmsff = returnscatter(feh_diff2f[take])
    biasa,rmsa = returnscatter(alpha_diff2[take])
    biasaf,rmsaf = returnscatter(alpha_diff2f[take])
    biasc,rmsc = returnscatter(c_diff2[take])
    biascf,rmscf = returnscatter(c_diff2f[take])
    biasn,rmsn = returnscatter(n_diff2[take])
    biasnf,rmsnf = returnscatter(n_diff2f[take])
    biaso,rmso = returnscatter(o_diff2[take])
    biasof,rmsof = returnscatter(o_diff2f[take])
    biasna,rmsna = returnscatter(na_diff2[take])
    biasnaf,rmsnaf = returnscatter(na_diff2f[take])
    biass,rmss = returnscatter(s_diff2[take])
    biassf,rmssf = returnscatter(s_diff2f[take])
    biasv,rmsv = returnscatter(v_diff2[take])
    biasvf,rmsvf = returnscatter(v_diff2f[take])
    biasmn,rmsmn = returnscatter(mn_diff2[take])
    biasmnf,rmsmnf = returnscatter(mn_diff2f[take])
    biasni,rmsni = returnscatter(ni_diff2[take])
    biasnif,rmsnif = returnscatter(ni_diff2f[take])
    biasmg,rmsmg = returnscatter(mg_diff2[take])
    biasmgf,rmsmgf = returnscatter(mg_diff2f[take])
    biasal,rmsal = returnscatter(al_diff2[take])
    biasalf,rmsalf = returnscatter(al_diff2f[take])
    biast_apogee, rmst_apogee = returnscatter(t_diff2_apogee[take])
    biasg_apogee,rmsg_apogee = returnscatter(g_diff2_apogee[take])
    biasf_apogee,rmsf_apogee = returnscatter(feh_diff2_apogee[take])
    biasa_apogee,rmsa_apogee = returnscatter(alpha_diff2_apogee[take])
    biasc_apogee,rmsc_apogee = returnscatter(c_diff2_apogee[take])
    biasna_apogee,rmsna_apogee = returnscatter(na_diff2_apogee[take])
    biaso_apogee,rmso_apogee = returnscatter(o_diff2_apogee[take])
    biasn_apogee,rmsn_apogee = returnscatter(n_diff2_apogee[take])
    biass_apogee,rmss_apogee = returnscatter(s_diff2_apogee[take])
    biasal_apogee,rmsal_apogee = returnscatter(al_diff2_apogee[take])
    biasmg_apogee,rmsmg_apogee = returnscatter(mg_diff2_apogee[take])
    biasmn_apogee,rmsmn_apogee = returnscatter(mn_diff2_apogee[take])
    biasv_apogee,rmsv_apogee = returnscatter(v_diff2_apogee[take])
    biasni_apogee, rmsni_apogee = returnscatter(ni_diff2_apogee[take])
    bias_t.append(biast)
    rms_t.append(rmst)
    rms_tf.append(rmstf)
    bias_g.append(biasg)
    rms_g.append(rmsg)
    rms_gf.append(rmsgf)
    bias_f.append(biasf)
    rms_f.append(rmsf)
    rms_ff.append(rmsff)
    bias_a.append(biasa)
    rms_a.append(rmsa)
    rms_af.append(rmsaf)
    bias_c.append(biasc)
    rms_c.append(rmsc)
    rms_cf.append(rmscf)
    bias_n.append(biasn)
    rms_n.append(rmsn)
    rms_nf.append(rmsnf)
    bias_o.append(biaso)
    rms_o.append(rmso)
    rms_of.append(rmsof)
    bias_mg.append(biasmg)
    rms_mg.append(rmsmg)
    rms_mgf.append(rmsmgf)
    bias_mn.append(biasmn)
    rms_mn.append(rmsmn)
    rms_mnf.append(rmsmnf)
    bias_v.append(biasv)
    rms_v.append(rmsv)
    rms_vf.append(rmsvf)
    bias_s.append(biass)
    rms_s.append(rmss)
    rms_sf.append(rmssf)
    bias_na.append(biasna)
    bias_al.append(biasal)
    rms_al.append(rmsal)
    rms_alf.append(rmsalf)
    rms_ni.append(rmsni)
    rms_nif.append(rmsnif)
    bias_ni.append(biasni)
    rms_na.append(rmsna)
    rms_naf.append(rmsnaf)
    bias_t_apogee.append(biast_apogee)
    rms_t_apogee.append(rmst_apogee)
    bias_g_apogee.append(biasg_apogee)
    rms_g_apogee.append(rmsg_apogee)
    bias_f_apogee.append(biasf_apogee)
    rms_f_apogee.append(rmsf_apogee)
    bias_a_apogee.append(biasa_apogee)
    rms_a_apogee.append(rmsa_apogee)
    bias_n_apogee.append(biasn_apogee)
    bias_ni_apogee.append(biasni_apogee)
    rms_ni_apogee.append(rmsni_apogee)
    rms_n_apogee.append(rmsn_apogee)
    bias_s_apogee.append(biass_apogee)
    rms_s_apogee.append(rmss_apogee)
    bias_c_apogee.append(biasc_apogee)
    rms_c_apogee.append(rmsc_apogee)
    bias_o_apogee.append(biaso_apogee)
    rms_o_apogee.append(rmso_apogee)
    rms_mg_apogee.append(rmsmg_apogee)
    rms_mn_apogee.append(rmsmn_apogee)
    rms_v_apogee.append(rmsv_apogee)
    rms_na_apogee.append(rmsna_apogee)
    bias_na_apogee.append(biasna_apogee)
    rms_al_apogee.append(rmsal_apogee)
    bias_al_apogee.append(biasal_apogee)

ax1.plot(snrmean, rms_t, 'ko')
ax1.plot(snrmean, rms_t, 'k-',label = 'The Cannon with filters')
ax1.plot(snrmean, rms_tf, 'b-',label = 'The Cannon sans filters')
ax1.plot(snrmean, rms_tf, 'bo')
ax2.plot(snrmean, rms_g, 'ko')
ax2.plot(snrmean, rms_g, 'k-')
ax2.plot(snrmean, rms_gf, 'b-')
ax2.plot(snrmean, rms_gf, 'bo')
ax3.plot(snrmean, rms_f, 'ko')
ax3.plot(snrmean, rms_f, 'k-')
ax3.plot(snrmean, rms_ff, 'b-')
ax3.plot(snrmean, rms_ff, 'bo')
ax4.plot(snrmean, rms_a, 'ko')
ax4.plot(snrmean, rms_a, 'k-')
ax4.plot(snrmean, rms_af, 'b-')
ax4.plot(snrmean, rms_af, 'bo')
ax5.plot(snrmean, rms_c, 'ko')
ax5.plot(snrmean, rms_c, 'k-')
ax5.plot(snrmean, rms_cf, 'b-')
ax5.plot(snrmean, rms_cf, 'bo')
ax6.plot(snrmean, rms_n, 'ko')
ax6.plot(snrmean, rms_n, 'k-')
ax6.plot(snrmean, rms_nf, 'b-')
ax6.plot(snrmean, rms_nf, 'bo')
ax7.plot(snrmean, rms_o, 'ko')
ax7.plot(snrmean, rms_o, 'k-')
ax7.plot(snrmean, rms_of, 'b-')
ax7.plot(snrmean, rms_of, 'bo')
ax8.plot(snrmean, rms_na, 'ko')
ax8.plot(snrmean, rms_na, 'k-')
ax8.plot(snrmean, rms_naf, 'b-')
ax8.plot(snrmean, rms_naf, 'bo')
ax9.plot(snrmean, rms_mg, 'ko')
ax9.plot(snrmean, rms_mg, 'k-')
ax9.plot(snrmean, rms_mgf, 'b-')
ax9.plot(snrmean, rms_mgf, 'bo')
ax10.plot(snrmean, rms_al, 'ko')
ax10.plot(snrmean, rms_al, 'k-')
ax10.plot(snrmean, rms_alf, 'b-')
ax10.plot(snrmean, rms_alf, 'bo')
ax10.plot(snrmean, rms_al, 'ko')
ax10.plot(snrmean, rms_al, 'ko')
ax11.plot(snrmean, rms_s, 'ko')
ax11.plot(snrmean, rms_s, 'k-')
ax11.plot(snrmean, rms_sf, 'b-')
ax11.plot(snrmean, rms_sf, 'bo')
ax12.plot(snrmean, rms_v, 'ko')
ax12.plot(snrmean, rms_v, 'k-')
ax12.plot(snrmean, rms_vf, 'b-')
ax12.plot(snrmean, rms_vf, 'bo')
ax13.plot(snrmean, rms_mn, 'ko')
ax13.plot(snrmean, rms_mn, 'k-')
ax13.plot(snrmean, rms_mnf, 'b-')
ax13.plot(snrmean, rms_mnf, 'bo')
ax14.plot(snrmean, rms_ni, 'ko')
ax14.plot(snrmean, rms_ni, 'k-')
ax14.plot(snrmean, rms_nif, 'b-')
ax14.plot(snrmean, rms_nif, 'bo')

ax1.plot(snrmean, rms_t_apogee, 'ro', linewidth = 1)
ax1.plot(snrmean, rms_t_apogee, 'r-', linewidth = 1, label = 'ASPCAP')
ax2.plot(snrmean, rms_g_apogee, 'ro', linewidth = 1)
ax2.plot(snrmean, rms_g_apogee, 'r-', linewidth = 1)
ax3.plot(snrmean, rms_f_apogee, 'ro', linewidth = 1)
ax3.plot(snrmean, rms_f_apogee, 'r-', linewidth = 1)
ax4.plot(snrmean, rms_a_apogee, 'ro', linewidth = 1)
ax4.plot(snrmean, rms_a_apogee, 'r-', linewidth = 1)
ax5.plot(snrmean, rms_c_apogee, 'ro', linewidth = 1)
ax5.plot(snrmean, rms_c_apogee, 'r-', linewidth = 1)
ax6.plot(snrmean, rms_n_apogee, 'ro', linewidth = 1)
ax6.plot(snrmean, rms_n_apogee, 'r-', linewidth = 1)
ax7.plot(snrmean, rms_o_apogee, 'ro', linewidth = 1)
ax7.plot(snrmean, rms_o_apogee, 'r-', linewidth = 1)
ax8.plot(snrmean, rms_na_apogee, 'ro', linewidth = 1)
ax8.plot(snrmean, rms_na_apogee, 'r-', linewidth = 1)
ax9.plot(snrmean, rms_mg_apogee, 'ro', linewidth = 1)
ax9.plot(snrmean, rms_mg_apogee, 'r-', linewidth = 1)
ax10.plot(snrmean, rms_al_apogee, 'ro', linewidth = 1)
ax10.plot(snrmean, rms_al_apogee, 'r-', linewidth = 1)
ax11.plot(snrmean, rms_s_apogee, 'ro', linewidth = 1)
ax11.plot(snrmean, rms_s_apogee, 'r-', linewidth = 1)
ax12.plot(snrmean, rms_v_apogee, 'ro', linewidth = 1)
ax12.plot(snrmean, rms_v_apogee, 'r-', linewidth = 1)
ax13.plot(snrmean, rms_mn_apogee, 'ro', linewidth = 1)
ax13.plot(snrmean, rms_mn_apogee, 'r-', linewidth = 1)
ax14.plot(snrmean, rms_ni_apogee, 'ro', linewidth = 1)
ax14.plot(snrmean, rms_ni_apogee, 'r-', linewidth = 1)
ax1.legend(fontsize = 7,frameon=False)
#axall = [ax1,ax2,ax3,ax4]

def mkn_set_ticks(ax, xticks):
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)

xmax = 220
for ax in [ax1,ax2,ax3,ax4,ax5,ax6, ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]:
  ax.set_xlim(0,xmax)
  #ax.set_xlabel("S/N of Visit Spectra", fontsize = f3, labelpad = 10 ) 
for ax in [ax5,ax6,ax7,ax8,ax9,ax10,ax11, ax12, ax13, ax14, ax15]:
    ax.set_ylim(0,0.4) 
ax1.set_ylim(0,200)
ax2.set_ylim(0,0.4)
ax3.set_ylim(0,0.3)
ax4.set_ylim(0,0.25)
f3 = 10
ax13.set_xlabel("S/N of Visit Spectra", fontsize = f3, labelpad = 10 ) 
ax3.xaxis.set_label_coords(1.05, -0.10)
fsize = 17
lpad = 7
lpad2 = 7 # -3
f3 =10
ax1.set_title("Teff (K) ", fontsize = f3)
ax2.set_title("log g (dex) ", fontsize = f3)
ax3.set_title("[Fe/H] (dex) ", fontsize = f3)
ax4.set_title(r"[$\alpha$/Fe] (dex) ", fontsize = f3)
ax5.set_title("[C/M]", fontsize = f3)
ax6.set_title("[N/M]", fontsize = f3)
ax7.set_title("[O/M]", fontsize = f3)
ax8.set_title("[Na/H]", fontsize = f3)
ax9.set_title("[Mg/M]", fontsize = f3)
ax10.set_title("[Al/H]", fontsize = f3)
ax11.set_title("[S/H]", fontsize = f3)
ax12.set_title("[V/H]", fontsize = f3)
ax13.set_title("[Mn/H]", fontsize = f3)
ax14.set_title("[Ni/H]", fontsize = f3)

f2 = 15.
ax6.set_ylabel("RMS: Combined $-$ Visit Spectra ", fontsize = f2, labelpad = 10 ) 
f.subplots_adjust(hspace=0.25)
f.subplots_adjust(bottom=0.10)
show()
draw()
savefig('14labels_zoom.pdf', bbox = 'tight', fmt = "pdf") 
