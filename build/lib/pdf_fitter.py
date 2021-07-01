from LNPLPL_functions import *
from LNPL_functions import *
import warnings

class Params():
    def __init__(self, root_finding_lower_lim= -15, root_finding_upper_lim=1, root_finding_step_size=0.2, \
                root_finding_max_tries=40, s_cut_off = 10, y_min_cut_off=0, bounds = [(0,1.01,0.3,3),(5,5,5,10)],\
                shrink_data = 4, debug=False, single_bounds=[(1,1),(5,5)]):
        
        ''' Parameters: 
        root_finding_lower_lim - This is the lower limit for the root finding range.
        root_finding_upper_lim - This is the upper limit for the root finding range.
        root_finding_step_size - This is the step_size for root finding. Range shrinks by this value after each iteration.
        root_finding_max_tries - This is the maximum number of attempts to make at root_finding.
        s_cut_off - The maximum cut-off for the x values.
        y_min_cut_off - The minimum cut-off for the y-values.
        bounds - The bounds of the parameters (double PL case).
        shrink_data - The number of data points to remove from the edges.
        debug - Flag for printing the parameter values while fitting is ongoing.
        single_bounds - The bounds of the fitted parameters (single PL case).
        '''
        self.root_finding_lower_lim=root_finding_lower_lim
        self.root_finding_upper_lim=root_finding_upper_lim
        self.root_finding_step_size=root_finding_step_size   #This means the range shrinks by 0.2 (default) everytime.
        self.root_finding_max_tries=root_finding_max_tries
        self.s_cut_off=s_cut_off
        self.y_min_cut_off=y_min_cut_off
        self.bounds = bounds
        self.shrink_data = shrink_data                       #Removing some data points from the edges helps to fit better.
        self.debug=debug
        self.single_bounds=single_bounds
        

        
class Result():
    def __init__(self, sigma=0, sigma_err=0, alpha=0, alpha_err=0, alpha1=0, \
                                  alpha1_err=0, sb=0, sb_err=0, st=0, s0=0, stbootmean=0, s0bootmean=0, st_err=0, s0_err=0):
        '''
        Container class for the result of the fit.
        Parameters:
        sigma, alpha_g, alpha_d, sg, sd, s0, sgbootmean, s0bootmean.
        For the errors, add _err to the end, e.g. sigma_err, sg_err, s0_err.
        Use sgbootmean and s0bootmean for the LN+2PL case. 
        The LN+2PL case spits out sg and s0 too but these are the values before the bootstrapping. 
        Use sg and s0 for the LN+PL case.
        Access these by doing result.sigma, result.alpha_g etc. 
        '''
        self.sigma=sigma
        self.sigma_err=sigma_err
        self.alpha_g=alpha
        self.alpha_g_err=alpha_err
        self.alpha_d=alpha1
        self.alpha_d_err=alpha1_err
        self.sd=sb
        self.sd_err=sb
        self.sg=st
        self.s0=s0
        self.sgbootmean=stbootmean
        self.s0bootmean=s0bootmean
        self.sg_err=st_err
        self.s0_err=s0_err
        
        

def PLFit(xdata, ydata, p0, params, use_K21=True, print_result=True):
    warnings.filterwarnings("ignore")
    index = np.where(ydata>params.y_min_cut_off)
    new_datapts = np.take(ydata, index[0])
    new_datapts_x = np.take(xdata, index[0])
    ydata = new_datapts
    xdata = new_datapts_x

    index = np.where(xdata<params.s_cut_off)
    new_datapts = np.take(ydata, index[0])
    new_datapts_x = np.take(xdata, index[0])
    ydata = new_datapts
    xdata = new_datapts_x

    if (use_K21==True):
        cdf = np.zeros(len(xdata))
        y = np.zeros(len(xdata))

        total = np.sum(ydata)
        for i in range(0, len(ydata)):
            cdf[i] = np.sum(ydata[0:i])/total
            y[i] = np.log(cdf[i]/(1-cdf[i]))  

        
        poptPL, pcov = curve_fit(LNPLfunctionCDFLog, xdata[params.shrink_data:-params.shrink_data], \
                                   y[params.shrink_data:-params.shrink_data],\
                                   p0=p0, bounds=params.single_bounds)
        perrPL = np.sqrt(np.diag(pcov))
        
        sigma = poptPL[0]
        alpha = poptPL[1]
        sigma_err = perrPL[0]
        alpha_err = perrPL[1]

        s0 = Finds0(sigma, alpha)
        st = s0 + alpha*sigma**2
        s0_err = Finds0err(sigma, sigma_err, alpha, alpha_err)
        st_err = FindstError(s0, alpha, sigma, s0_err, alpha_err, sigma_err)   

        if print_result==True:
            print ('//============Fit complete=========//')
            print ('Best fit parameters: \n', 'sigma=',sigma,'+/-', sigma_err, '\n',\
                  'alpha=', alpha,'+/-', alpha_err, '\n', 's0=',s0,'+/-', s0_err, '\n', 'st=',st,'+/-', st_err)


        result = Result(sigma=sigma, sigma_err=sigma_err, alpha=alpha, alpha_err=alpha_err, st=st, \
                            s0=s0, st_err=st_err, s0_err=s0_err)

        return result

    ######
    #else: WRITE YOUR OWN CODE HERE.



def PLPLFit(xdata, ydata, p0, params, use_K21=True, print_result=True):
    '''
    This function will return the fitted parameters directly. 
    Params: 
    xdata - the x values of the data
    ydata - the y values of the data
    p0 - the initial guess for fitting
    print_result - Set to True by default. To remove the output on the screen, add print_result=False to function call. 
    '''
    
    fit = PLPLfit_func(params) 
    
    index = np.where(ydata>fit.y_min_cut_off)
    new_datapts = np.take(ydata, index[0])
    new_datapts_x = np.take(xdata, index[0])
    ydata = new_datapts
    xdata = new_datapts_x
    
    index = np.where(xdata<fit.s_cut_off)
    new_datapts = np.take(ydata, index[0])
    new_datapts_x = np.take(xdata, index[0])
    ydata = new_datapts
    xdata = new_datapts_x
    
    
    if (use_K21==True):
        cdf = np.zeros(len(xdata))
        y = np.zeros(len(xdata))

        total = np.sum(ydata)
        warnings.filterwarnings("ignore")
        for i in range(0, len(ydata)):
            cdf[i] = np.sum(ydata[0:i])/total
            y[i] = np.log(cdf[i]/(1-cdf[i])) 
    
        
        
        poptPLPL, pcov = curve_fit(fit.LNPLPLfunctionCDFLog, xdata[fit.shrink_data:-fit.shrink_data], \
                                   y[fit.shrink_data:-fit.shrink_data],\
                               p0 = p0, bounds=fit.bounds, maxfev=50000)

        perrPLPL = np.sqrt(np.diag(pcov))
        
        
        sigma = poptPLPL[0]
        alpha = poptPLPL[1]
        sigma_err = perrPLPL[0]
        alpha_err = perrPLPL[1]
        alpha1 = poptPLPL[2]
        alpha1_err = perrPLPL[2]
        sb = poptPLPL[3]
        sb_err = perrPLPL[3]

        
        s0 = fit.FindTheFirstRootPlease(sigma, alpha, alpha1, sb, fit.root_finding_upper_lim, 1)

        st = s0+alpha*sigma**2
        s0bootmean, s0_err, stbootmean, st_err = fit.BootStapper(poptPLPL, perrPLPL)
        
        if print_result==True:
            print ('//============Fit complete=========//')
            print ('Best fit parameters: \n', 'sigma=',sigma,'+/-', sigma_err, '\n',\
                  'alpha=', alpha,'+/-', alpha_err, '\n', 'alpha1=',alpha1,'+/-', alpha1_err, '\n', \
                   'sb=',sb,'+/-', sb_err)
            print ('s0, st = ', s0, st)
            print ('After bootstrapping: s0, st=', s0bootmean,'+/-', s0_err, stbootmean, '+/-', st_err)
        
        result = Result(sigma=sigma, sigma_err=sigma_err, alpha=alpha, alpha_err=alpha_err, alpha1=alpha1, \
                                  alpha1_err=alpha1_err, sb=sb, sb_err=sb_err, st=st, \
                        s0=s0, stbootmean=stbootmean, s0bootmean=s0bootmean, st_err=st_err, s0_err=s0_err)
        
        return result
    
    ######
    #else: WRITE YOUR OWN CODE HERE.
    
    
    