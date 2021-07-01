from pdf_fitter import *
import numpy as np


sample_data = np.load('./sample_data.npz')
xdata = sample_data['arr_0']
ydata = sample_data['arr_1']
sink = float(sample_data['arr_2'])


#fit = PLPLfit_func(params) 
#print ('Test...', fit.LNPLPLfunctionCDFLog(xdata, 1.5, 1.7, 1.2, 7.1))

params = Params(root_finding_lower_lim= -15, root_finding_upper_lim=1, root_finding_step_size=0.2, \
                root_finding_max_tries=40, s_cut_off = sink)

p0 = [1.7, 1.6, 0.8, 7.1]
PLPLresult = PLPLFit(xdata, ydata, p0, params, use_K21=True)

p0 = [1.85, 1.57]
PLresult = PLFit(xdata, ydata, p0, params)

print (PLresult.sigma_err, PLresult.alpha_g_err, PLPLresult.sigma_err, PLPLresult.sd)

