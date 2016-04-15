#  labels = ["teff", "logg", "feh", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Ti", "V", "Mn", "Ni", "P", "Cr", "Co", "Cu", "Rb"]
import pickle 
#if shape(b) == "None":
#a = open('self_2nd_order_tags_dr13_nofilt.pickle', 'r')
#a = open('input_outputs_outputscaled_filt.pickle') 
a = open('input_outputs_outputscaled_filtA.pickle') 
a = open('input_outputs_outputscaled_filtB.pickle') 
a = open('input_outputs_test.pickle') 
#a = open('input_outputs_outputscaled_filtB_nofilt.pickle') 
#a = open('input_outputs_outputscaled_nofiltcode.pickle') 
b = pickle.load(a)
a.close()
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

fn = "training_dr13e2_large_test.list"
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

#outputs_rescaled = b[2]
outputs= b[1]
outputs_rescaled = outputs 
inputs = [T_est, g_est, feh_est, C, N, O, Na, Mg, Al, Si, S, K, Ca, Ti, V, Mn, Ni, P, Cr, Co, Cu, Rb]
#outputs_rescaled[2] = outputs[2]
#rangeit = arange(0,loutputs,1)
#for a,b,c in zip(inputs, outputs, rangeit):
#outputs_rescaled = [] 
#for each in rangeit:
#  #outputs[c] = unscale(outputs[c], inputs[c]) 
#  valout = unscale(outputs[each], inputs[each]) 
#  #valout = unscale(outputs[:,each], inputs[each]) 
#  outputs_rescaled.append(valout)
###
