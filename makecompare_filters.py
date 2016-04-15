import pickle 
import pyfits 
def rescale(in_array):
  valin = percentile(in_array, (50, 84))
  valscale = (valin[1]-valin[0])/2
  valoff = (valin[0])
  scaled_val = (in_array - valoff) / valscale
  return scaled_val
def unscale(scaled_val, in_array):
  valin = percentile(in_array, (50, 84))
  valscale = (valin[1]-valin[0])/2
  valoff = valin[0]
  out_val = scaled_val*valscale  + valoff 
  return out_val

#a = pyfits.open('/Users/ness/new_laptop/whitepaper/cal.fits')
a = pyfits.open('cal.fits')
apid = a[1].data['APOGEE_ID']
snr = a[1].data['SNR'] 
fparam = a[1].data['FPARAM']
felem = a[1].data['FELEM']
ch, nh = felem[:,0], felem[:,9] 
ch, nh = felem[:,0], felem[:,9] 
alin, cain, fein, kin  = felem[:,1], felem[:,2], felem[:,3], felem[:,4]
mgin, mnin, nain, niin  = felem[:,5], felem[:,6], felem[:,7], felem[:,8]
nin, oin, siin, sinval  = felem[:,9], felem[:,10], felem[:,11], felem[:,12]
tiin, vin  = felem[:,13], felem[:,14]
tapogee = fparam[:,0]
gapogee = fparam[:,1]
fehapogee = fparam[:,3]
alphaapogee = fparam[:,-1]
tin,gin,fehin,alphain = fparam[:,0], fparam[:,1], fparam[:,3], fparam[:,-1]
ch, nh = felem[:,0], felem[:,9] 
fehval = felem[:,3] 
nfe = nh #- fehval
cfe = ch #- fehval# the SNR 290 is trained on C/H and N/H 
cin,nin = cfe, nfe 
field = a[1].data['FIELD']

unique_ids = unique(apid)
filein2 = open("cal_file_tags_dr13_filt_30e.pickle", 'r') 
b = pickle.load(filein2) 
filein2.close()

fn = 'training_dr13e2_large.list'
fn = 'training_dr13e2_large_test.list'
T_est,g_est,feh_est,alpha_est, T_A, g_A, feh_A,rc_est = np.loadtxt(fn, usecols = (1,2,3,4,1,2,3,4), unpack =1) 
C, N, O, Na, Mg, Al, Si, P, S, K, Ca, Ti, V, Cr, Mn,Co, Fe, Ni, Cu, Rb = loadtxt(fn, usecols = (5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23, 26), unpack =1)
T_est,g_est,feh_est,alpha_est, T_A, g_A, feh_A,rc_est = np.loadtxt(fn, usecols = (1,2,3,4,1,2,3,4), unpack =1) 
C, N, O, Na, Mg, Al, Si, P, S, K, Ca, Ti, V, Cr, Mn,Co, Fe, Ni, Cu,Rb = loadtxt(fn, usecols = (5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26), unpack =1)
offsets = np.array([mean(rescale(T_est)), mean(rescale(g_est)), mean(rescale(feh_est)),mean(rescale(C)), mean(rescale(N)), mean(rescale(O)), mean(rescale(Na)), mean(rescale(Mg)), mean(rescale(Al)), mean(rescale(Si)), mean(rescale(S)), mean(rescale(K)), mean(rescale(Ca)), mean(rescale(Ti)), mean(rescale(V)), mean(rescale(Mn)),mean(rescale(Ni)), mean(rescale(P)), mean(rescale(Cr)), mean(rescale(Co)), mean(rescale(Cu)), mean(rescale(Rb))]) 
labels = ["teff", "logg", "feh", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Ti", "V", "Mn", "Ni", "P", "Cr", "Co", "Cu", "Rb"]
T_est_rs = rescale(T_est)
g_est_rs = rescale(g_est)
feh_est_rs = rescale(feh_est)
alpha_est_rs = rescale(alpha_est)
C_rs, N_rs, O_rs, Na_rs, Mg_rs, Al_rs = rescale(C), rescale(N), rescale(O), rescale(Na), rescale(Mg), rescale(Al) 
Si_rs, S_rs, K_rs, Ca_rs, Ti_rs, V_rs, Mn_rs, Fe_rs, Ni_rs = rescale(Si), rescale(S), rescale(K), rescale(Ca), rescale(Ti), rescale(V), rescale(Mn), rescale(Fe), rescale(Ni) 
P_rs, Cr_rs, Co_rs, Rb_rs = rescale(P), rescale(Cr), rescale(Co), rescale(Rb) 
Cu_rs = rescale(Cu) 
P_rs, Cr_rs, Co_rs, Rb_rs = rescale(P), rescale(Cr), rescale(Co), rescale(Rb) 
Cu_rs = rescale(Cu) 

outputs = b[0] 

chi = b[2]
chi2_val = sum(chi[:,:]*chi[:,:], axis=0)
nums_a = []
len1 = shape(chi)[1]
for i in range(0,len1):
  nums_a.append(len(chi[:,i][chi[:,i] == 0.]) )
nums_a = array(nums_a)
nums = shape(chi)[0] - nums_a
chi2_dof = chi2_val*1./nums

outvals = []
diffvals = [] 
for inval, outval in zip(labels, outputs):
    outval2 = unscale(outval, inval) 
    outvals.append(unscale(outval, inval)) 
outvals = array(outvals) 



labels_str = ["teff", "logg", "feh","alpha", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Ti", "V", "Mn", "Ni"]
tout, gout, fehout = outvals[0], outvals[1], outvals[2] 
alpha, Cout, Nout, Oout, Naout, Mgout, Alout, Siout, Sout,Kout, Caout=  outvals[3], outvals[4], outvals[5], outvals[6], outvals[7], outvals[8], outvals[9], outvals[10], outvals[11], outvals[12], outvals[13]
Tiout, Vout, Mnout, Niout= outvals[13], outvals[14], outvals[15], outvals[16]

labels_str = ["teff", "logg", "feh", "alpha", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Ti", "V", "Mn", "Ni"]
T_est,g_est,feh_est,alpha_est, T_A, g_A, feh_A,rc_est = np.loadtxt(fn, usecols = (1,2,3,4,1,2,3,4), unpack =1) 
Cout, Nout, Alout, Mgout, Naout, Oout, Sout, Vout,Mnout, Niout =  outvals[3], outvals[4], outvals[5], outvals[6], outvals[7], outvals[8], outvals[9], outvals[10], outvals[11], outvals[12]






params_highsnr = []
params_unique = []
chi2_uniquef =[]
field_unique = []
params_elem_unique = []
id_take = [] 
snr_take = [] 
apogee_unique = [] 
apogee_felem_unique = [] 
apogee_id_unique = [] 
unique_ids = array(unique_ids) 
cin = array(cin)
nin = array(nin) 
field = array(field) 
apogee_id_take = array(field) 
for each in unique_ids:
    take = apid == each
    params_take = outvals.T[take] 
    chi2_take = chi2_dof[take] 
    apogee_take = fparam[take] 
    apogee_felem_take = felem[take] 
    apogee_id_take = apid[take] 
    tin_take = tin[take]
    gin_take = gin[take]
    alphain_take = alphain[take]
    field_take = field[take]
    fehin_take = fehin[take]
    cin_take = cin[take]
    nin_take = nin[take]
    alin_take = alin[take]
    mgin_take = mgin[take]
    mnin_take = mnin[take]
    alin_take = alin[take]
    vin_take = vin[take]
    niin_take = niin[take]
    nain_take = nain[take]
    sin_take = sinval[take]
    oin_take = oin[take]
    snrtake = snr[take] 
    chi2take = chi2_dof[take] 
    apidtake = apid[take] 
    snr_highest = argsort(snr[take]) 
    if len(take[take]) > 1:
      print len(take[take]) 
      t_test = logical_and(tin_take[0] > 3500, tin_take[0] < 6500)
      g_test = logical_and(gin_take[0] > -2, gin_take[0] < 3.6)
      alpha_test = logical_and(alphain_take[0] > -2, alphain_take[0] < 1)
      feh_test = logical_and(fehin_take[0] > -1.6, fehin_take[0] < 1.4)
      c_test = logical_and(cin_take[0] > -12.8, cin_take[0] < 31.4)
      n_test = logical_and(nin_take[0] > -12.8, nin_take[0] < 31.4)
      o_test = logical_and(oin_take[0] > -12.8, oin_take[0] < 31.4)
      na_test = logical_and(nain_take[0] > -12.8, nain_take[0] < 31.4)
      mg_test = logical_and(mgin_take[0] > -12.8, mgin_take[0] < 31.4)
      al_test = logical_and(alin_take[0] > -12.8, alin_take[0] < 31.4)
      s_test = logical_and(sin_take[0] > -12.8, sin_take[0] < 31.4)
      v_test = logical_and(vin_take[0] > -12.8, vin_take[0] < 31.4)
      mn_test = logical_and(mnin_take[0] > -12.8, mnin_take[0] < 31.4)
      ni_test = logical_and(niin_take[0] > -12.8, niin_take[0] < 31.4)
      vsnio_test = logical_and(logical_and(v_test, s_test), logical_and(s_test, ni_test)) 
      mnmgal_test = logical_and(logical_and(mg_test, mn_test), logical_and(logical_and(al_test, ni_test), o_test) ) 
      n_test = logical_and(logical_and(vsnio_test, c_test), logical_and(al_test, n_test))
      feh_test = logical_and(mnmgal_test, logical_and(n_test, feh_test) ) 
      if logical_and(logical_and(alpha_test, feh_test), logical_and(t_test, g_test)):
          params_highsnr.append(params_take[0] )
          params_unique.append(params_take)
          params_elem_unique.append(params_take)
          field_unique.append(field_take)
          apogee_unique.append(apogee_take)
          apogee_felem_unique.append(apogee_felem_take)
          apogee_id_unique.append(apogee_id_take)
          id_take.append(apidtake) 
          snr_take.append(snrtake) 
          chi2_uniquef.append(chi2take) 
     
t_diff = []
g_diff = []
feh_diff = []
alpha_diff = [] 
t_diff_apogee = []
g_diff_apogee = []
feh_diff_apogee = []
c_diff_apogee = []
n_diff_apogee = []
al_diff_apogee = []
alpha_diff_apogee = [] 
t_val, g_val, feh_val, alpha_val = [],[],[],[]
snr_val = [] 
chi_valf = []
snr_max = []
fehall = []
fehallap = []
c_diff = []
al_diff = []
al_diff_apogee = []
mg_diff = []
mg_diff_apogee = []
mn_diff = []
mn_diff_apogee = []
na_diff = []
na_diff_apogee = []
s_diff = []
s_diff_apogee = []
v_diff = []
v_diff_apogee = []
ni_diff = []
ni_diff_apogee = []
o_diff = []
o_diff_apogee = []
c_apogee = []
na_apogee = []
na_apogee_max = []
n_apogee = []
c,n = [],[]
t_apogee = []
g_apogee = [] 
n_diff = []
t_highsnr = []
g_highsnr = []
feh_highsnr = []
alpha_highsnr = []
mg_highsnr = []
mg_highsnr_ap = []
feh_highsnr_ap = []
alpha_highsnr_ap = []
t_highsnr_ap = []
g_highsnr_ap = []
alpha_highsnr = []
alpha_highsnr_ap = []
c_highsnr = []
n_highsnr = []
o_highsnr = []
na_highsnr = []
na_highsnr_ap = []
mn_highsnr_ap = []
mg_highsnr_ap = []
mn_highsnr = []
al_highsnr= []
s_highsnr= []
o_highsnr= []
ni_highsnr= []
mn_highsnr= []
v_highsnr= []

c_highsnr_ap = []
n_highsnr_ap = []
field_highsnr = [] 
ids_highsnr = [] 
id_val = [] 
for each,snrval,chival,apeach,apeach_felem, each_elem, each_field,idval in zip(params_unique, snr_take, chi2_uniquef, apogee_unique, apogee_felem_unique, params_elem_unique, field_unique, id_take):
    chi_valf.append(chival[1:])
    #id_val.append([idval]*(len(each)-1))
    id_val.append([idval[0]]*(len(each)-1))
    #id_val.append([idval[0]]*len(chival[1:]))
    t_apogee.append( ( apeach[:,0])[1:])
    t_diff.append( (each[:,0][0] - each[:,0])[1:])
    t_diff_apogee.append(  (apeach[:,0][0] - apeach[:,0])[1:])
    t_highsnr_ap.append(  apeach[:,0][0])
    g_highsnr_ap.append(  apeach[:,1][0])
    t_val.append(  (each[:,0])[1:])
    g_diff.append(  (each[:,1][0] - each[:,1])[1:])
    g_apogee.append(( apeach[:,1])[1:])
    g_diff_apogee.append(  (apeach[:,1][0] - apeach[:,1])[1:])
    g_val.append(  (each[:,1])[1:] )
    field_highsnr.append( (each_field[0]))
    ids_highsnr.append( (idval[0]))
    g_highsnr.append(  (each[:,1])[0] )
    t_highsnr.append(  (each[:,0])[0] )
    feh_highsnr.append(  (each[:,2])[0] )
    alpha_highsnr.append(  (each[:,3])[0] )
    c_highsnr.append(  (each[:,4])[0] )
    n_highsnr.append(  (each[:,5])[0] )
    al_highsnr.append(  (each[:,6])[0] )
    mg_highsnr.append(  (each[:,7])[0] )
    na_highsnr.append(  (each[:,8])[0] )
    o_highsnr.append(  (each[:,9])[0] )
    s_highsnr.append(  (each[:,10])[0] )
    v_highsnr.append(  (each[:,11])[0] )
    mn_highsnr.append(  (each[:,12])[0] )
    ni_highsnr.append(  (each[:,13])[0] )
    feh_diff.append( ( each[:,2][0] - each[:,2])[1:])
    feh_diff_apogee.append( ( apeach[:,3][0] - apeach[:,3])[1:])
    feh_val.append(  (each[:,2])[1:] )
    alpha_diff.append( ( each[:,3][0] - each[:,3])[1:])
    alpha_diff_apogee.append( ( apeach[:,-1][0] - apeach[:,-1])[1:])
    alpha_val.append(  (each[:,3])[1:])
    snr_val.append( snrval[1:]) 
    snr_max.append( snrval[0]) 
    fehall.append(each[:,2][1:])
    fehallap.append(apeach[:,3][1:])
    feh_highsnr_ap.append(apeach[:,3][0])
    #n_diff_apogee.append(  (apeach_felem[:,9][0] - apeach_felem[:,9])[1:] -apeach[:,3][0] + apeach[:,3][1:])
    #c_diff_apogee.append(  (apeach_felem[:,0][0] - apeach_felem[:,0])[1:] - apeach[:,3][0] + apeach[:,3][1:])
    n_diff_apogee.append(  (apeach_felem[:,9][0] - apeach_felem[:,9])[1:]) 
    c_diff_apogee.append(  (apeach_felem[:,0][0] - apeach_felem[:,0])[1:])
    al_diff_apogee.append(  (apeach_felem[:,1][0] - apeach_felem[:,1])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    mg_diff_apogee.append(  (apeach_felem[:,5][0] - apeach_felem[:,5])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    mn_diff_apogee.append(  (apeach_felem[:,6][0] - apeach_felem[:,6])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    na_diff_apogee.append(  (apeach_felem[:,7][0] - apeach_felem[:,7])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    ni_diff_apogee.append(  (apeach_felem[:,8][0] - apeach_felem[:,8])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    o_diff_apogee.append(  (apeach_felem[:,10][0] - apeach_felem[:,10])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    s_diff_apogee.append(  (apeach_felem[:,12][0] - apeach_felem[:,12])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    v_diff_apogee.append(  (apeach_felem[:,14][0] - apeach_felem[:,14])[1:])# - apeach[:,3][0] + apeach[:,3][1:])
    c_apogee.append(  apeach_felem[:,0][1:] )
    n_apogee.append(  apeach_felem[:,9][1:] )
    na_apogee.append(  apeach_felem[:,7][1:] )
    na_apogee_max.append(  apeach_felem[:,7][0] )
    c_diff.append( (each[:,3][0] - each[:,3])[1:])
    n_diff.append( (each[:,4][0] - each[:,4])[1:])
    al_diff.append( (each[:,5][0] - each[:,5])[1:])
    mg_diff.append( (each[:,6][0] - each[:,6])[1:])
    na_diff.append( (each[:,7][0] - each[:,7])[1:])
    o_diff.append( (each[:,8][0] - each[:,8])[1:])
    s_diff.append( (each[:,9][0] - each[:,9])[1:])
    v_diff.append( (each[:,10][0] - each[:,10])[1:])
    mn_diff.append( (each[:,11][0] - each[:,11])[1:])
    ni_diff.append( (each[:,12][0] - each[:,12])[1:])

    c.append( (each[:,4])[1:])
    n.append( (each[:,5])[1:])
    c_highsnr_ap.append(  apeach_felem[:,0][0] )
   # alpha_highsnr_ap.append(  apeach_fparam[:,-1][0] )
    n_highsnr_ap.append(  apeach_felem[:,9][0] )
    o_highsnr_ap.append(  apeach_felem[:,10][0] )
    na_highsnr_ap.append(  apeach_felem[:,7][0] )
    mn_highsnr_ap.append(  apeach_felem[:,6][0] )
    mg_highsnr_ap.append( apeach_felem[:,5][0])

na_highsnr = array(na_highsnr)
na_highsnr_ap = array(na_highsnr_ap)
fehall = hstack((fehall))
fehallap = hstack((fehallap))
t_diff2, g_diff2, feh_diff2, alpha_diff2, snr_val2 = hstack((t_diff)), hstack((g_diff)), hstack((feh_diff)), hstack((alpha_diff)), hstack((snr_val)) 
t_val2, g_val2, feh_val2, alpha_val2 = hstack((t_val)), hstack((g_val)), hstack((feh_val)), hstack((alpha_val)) 
t_diff2_apogee, g_diff2_apogee, feh_diff2_apogee, alpha_diff2_apogee = hstack((t_diff_apogee)), hstack((g_diff_apogee)), hstack((feh_diff_apogee)), hstack((alpha_diff_apogee)) 
c_diff2_apogee, n_diff2_apogee = hstack((c_diff_apogee)), hstack((n_diff_apogee))
ni_diff2_apogee, ni_diff2 = hstack((ni_diff_apogee)), hstack((ni_diff))
mg_diff2_apogee, mg_diff2 = hstack((mg_diff_apogee)), hstack((mg_diff))
o_diff2_apogee, o_diff2 = hstack((o_diff_apogee)), hstack((o_diff))
na_diff2_apogee, na_diff2 = hstack((na_diff_apogee)), hstack((na_diff))
s_diff2_apogee, s_diff2 = hstack((s_diff_apogee)), hstack((s_diff))
v_diff2_apogee, v_diff2 = hstack((v_diff_apogee)), hstack((v_diff))
mn_diff2_apogee, mn_diff2 = hstack((mn_diff_apogee)), hstack((mn_diff))
al_diff2_apogee, al_diff2 = hstack((al_diff_apogee)), hstack((al_diff))
chi_val2f = hstack((chi_valf)) 
id_val2 = hstack((id_val)) 
snr_max2 = hstack((snr_max))
c_diff2 = hstack((c_diff))
n_diff2 = hstack((n_diff))
c = hstack((c))
n = hstack((n))
#c_apogee = hstack((c_apogee))
#na_apogee = hstack((na_apogee))
#n_apogee = hstack((n_apogee))

