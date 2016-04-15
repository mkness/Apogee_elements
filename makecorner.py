#import corner
#tdiff = diffvals[0]
#tdiff_nf = diffvals_nf[0]
#gdiff = diffvals[1]
#gdiff_nf = diffvals_nf[1]
#fehdiff = diffvals[2]
#fehdiff_nf = diffvals_nf[2]
#cdiff = diffvals[3]
#cdiff_nf = diffvals_nf[3]
#ndiff = diffvals[4]
#ndiff_nf = diffvals_nf[4]
#data = zip(tdiff[abs(tdiff) < 200],gdiff[abs(gdiff) < 0.4],fehdiff[abs(fehdiff) < 0.4],cdiff[abs(cdiff) < 0.4], ndiff[abs(ndiff) < 0.4]) 
#data = zip(tdiff_nf[abs(tdiff_nf) < 200],gdiff_nf[abs(gdiff_nf) < 0.4],fehdiff_nf[abs(fehdiff_nf) < 0.4],cdiff_nf[abs(cdiff_nf) < 0.4], ndiff_nf[abs(ndiff_nf) < 0.4]) 

labels = ["teff", "logg", "feh", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca", "Ti", "V", "Mn", "Ni", "P", "Cr", "Co", "Cu", "Rb", "blank", "blank"]
data = inputs 
data = array(data).T
figure = corner.corner(data,bins=15, labels=[r"${\Delta}T_{eff}$", r"${\Delta}logg$", r"${\Delta}[Fe/H]$",
                                     r"${\Delta}[C/H]$", r"${\Delta}$ [N/H]"],
                         show_titles=True, title_args={"fontsize": 12},smooth=0.5,scale_hist=True)
figure.gca().annotate("Covariances of Filtered Version", xy=(0.5, 1.0), xycoords="figure fraction",
                      xytext=(0, -5), textcoords="offset points",
                      ha="center", va="top")
figure.savefig("covs_filtered.png")
