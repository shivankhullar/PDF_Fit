import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import curve_fit


# ==================================================================== #
# ==================================================================== #
# ============== For the Lognormal + 2 Power-Law model =============== #
# ==================================================================== #
# ==================================================================== #

### Note that we use a different convention here. 
### The variables alpha, st, alpha_1, s_b are alpha_g, s_g, alpha_d and s_d in Khullar et al. (2021) respectively. 

### Find p0
def FindPLPLP0(st, sigma, alpha, s0, alpha1, sb):
    ''' This function returns the value of p_0. See equation A10 in Khullar et al. (2021). '''
    num = -np.exp(alpha*st-(st-s0)**2/(2*sigma**2))*(s0-st)
    den = alpha*np.sqrt(2*np.pi)*sigma**3
    return num/den

### Find p1
def FindPLPLP1(st, sigma, alpha, s0, alpha1, sb):
    ''' This function returns the value of p_1. See equation A12 in Khullar et al. (2021). '''
    num = -np.exp((alpha1-alpha)*sb+alpha*st-(st-s0)**2/(2*sigma**2))*(s0-st)
    den = alpha*np.sqrt(2*np.pi)*sigma**3
    return num/den
    
### Find Normalization when using -inf to +inf
def FindPLPLN(st, sigma, alpha, p0, s0, alpha1, sb, p1):
    ''' This function returns the value of N, the normalization constant. 
    This is not used since it is not calculated using a cut-off value. '''
    t1 = (-np.exp(-alpha*sb) + np.exp(-alpha*st))*p0/(alpha)
    t2 = np.exp(-alpha1*sb)*p1/(alpha1)
    t3 = 0.5*scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma))
    return 1.0/(t1+t2+t3)

### Find Normalization when using -inf to scut -- a max density cutoff defined to be sink creation threshold
def FindPLPLNscut(st, sigma, alpha, p0, s0, alpha1, sb, p1, scut):
    ''' This function returns the value of N, the normalization constant. 
    This is calculated using a cut-off value. See equation A9 in Khullar et al. (2021). '''
    t1 = (-np.exp(-alpha*sb) + np.exp(-alpha*st))*p0/(alpha)
    t2 = (np.exp(-alpha1*sb) - np.exp(-alpha1*scut))*p1/(alpha1)
    t3 = 0.5*scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma))
    return 1.0/(t1+t2+t3)

   

def s0rootfuncnew(s0, sigma, alpha, alpha1, sb, scut):
    ''' This is the function used for finding the value of s_0. It is the function that the root finding function uses.
    See equation A13 and Appendix A2 in Khullar et al. (2021) for more info on the whole procedure.'''
    if alpha==1.0 or alpha1==1.0:
        if alpha==1.0:
            alpha=1.001
            print ('Function blows up at alpha=1, adding 0.001...')
        else:
            alpha1=1.001
            print ('Function blows up at alpha1=1, adding 0.001...')
    st = s0 + alpha*sigma**2
    
    p0 = FindPLPLP0(st, sigma, alpha, s0, alpha1, sb)
    p1 = FindPLPLP1(st, sigma, alpha, s0, alpha1, sb)
    N = FindPLPLNscut(st, sigma, alpha, p0, s0, alpha1, sb, p1, scut)
    #N = FindPLPLN(st, sigma, alpha, p0, s0, alpha1, sb, p1)
    
    a = -2*np.exp(sb-alpha*sb)*p0/(alpha-1)
    b = 2*np.exp(st-alpha*st)*p0/(alpha-1)
    c = 2*np.exp(sb-alpha1*sb)*p1/(alpha1-1)
    d = -2*np.exp(scut-alpha1*scut)*p1/(alpha1-1)
    e = np.exp(s0+sigma**2/2.0)*scipy.special.erfc((s0+sigma**2-st)/(np.sqrt(2)*sigma))
    
    y = (a+b+c+d+e)*N/2.0 - 1 ## -1 for root finding
    
    return y


def FindstError(s0, alpha, sigma, s0_err, alpha_err, sigma_err):
    ''' This is the function used to find the error in the estimate of s_t (s_g in Khullar et al. (2021)). 
    Does so by using the quadrature rule to estimate errors. '''
    st = s0+alpha*sigma**2
    f = alpha*sigma**2
    df = abs(f)*np.sqrt((alpha_err/alpha)**2+(2*sigma_err/sigma))
    ds0 = s0_err
    dst = np.sqrt(ds0**2+df**2) 
    return dst


class PLPLfit_func(object):
    def __init__(self, object):
        
        ## Copying the parameters to to this class basically. Could have just used object everywhere but oh well..
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
        self.root_finding_lower_lim=object.root_finding_lower_lim
        self.root_finding_upper_lim=object.root_finding_upper_lim
        self.root_finding_step_size=object.root_finding_step_size   #This means the range shrinks by 0.2 (default) everytime.
        self.root_finding_max_tries=object.root_finding_max_tries
        self.s_cut_off=object.s_cut_off
        self.y_min_cut_off=object.y_min_cut_off
        self.bounds = object.bounds
        self.shrink_data = object.shrink_data                  #Removing some data points from the edges helps to fit better.
        self.debug=object.debug
        
        
    def FindTheFirstRootPlease(self, sigma, alpha, alpha1, sb, upper, n):
        ''' This is the function for finding the value of s_0 through root finding. It is recursive and tries to find
        the root by shrinking the range. See Appendix A2 in Khullar et al. (2021) for more info on the whole procedure.'''
        scut = self.s_cut_off
        ## n is the tracker for number of recursions/tries
        if n<self.root_finding_max_tries:
            try:
                s0 = optimize.bisect(lambda x: s0rootfuncnew(x, sigma, alpha, alpha1, sb, scut), \
                                     self.root_finding_lower_lim, upper)
                #print ('Root Found!', s0)
                return s0
            except ValueError:
                if n<20:
                    print ('Try %d: Trying root finding in smaller range, upper lim='%(n), upper, sigma, alpha, alpha1, sb)
                    s0 = FindTheFirstRootPlease(sigma, alpha, alpha1, sb, \
                                                upper-self.root_finding_step_size, n+1)
                if n>=20:
                    print ('Try %d: Root finding is not easy, upper lim='%(n), upper, sigma, alpha, alpha1, sb)
                    s0 = FindTheFirstRootPlease(sigma, alpha, alpha1, sb, \
                                                upper-self.root_finding_step_size, n+1)
        else:
            raise ValueError('Max iterations reached to find root')

        return s0
        
     
        
    def LNPLPLfunction(self, s, sigma, alpha, alpha1, sb):
        ''' This is the function that describes the LN+2PL function. See equation 8 in Khullar et al. (2021). 
        Returns p(s). '''
        upper = self.root_finding_upper_lim
        scut = self.s_cut_off
        s0 = FindTheFirstRootPlease(sigma, alpha, alpha1, sb, upper, 1)
        st = s0 + alpha*sigma**2
        if self.debug==True:
            print (sigma, alpha, alpha1, sb, st, s0)
        if (st>sb):
            print ('Skipping this case, st>sb ...')
            return np.ones(len(s))
        if (alpha1>alpha):
            print ('Skipping this case, alpha1>alpha ...')
            return np.ones(len(s))
        
        # Divide the x-range into 3 separate ranges for LN, 1st PL and 2nd PL.    
        s_list = np.where(s<st)[0]
        s_low = np.take(s, s_list)

        s_list = np.where((s>=st)&(s<sb))[0]
        s_high = np.take(s, s_list)

        s_list = np.where(s>=sb)[0]
        s_high2 = np.take(s, s_list)

        p0 = FindPLPLP0(st, sigma, alpha, s0, alpha1, sb)
        p1 = FindPLPLP1(st, sigma, alpha, s0, alpha1, sb)
        #N = FindPLPLN(st, sigma, alpha, p0, s0, alpha1, sb, p1)
        N = FindPLPLNscut(st, sigma, alpha, p0, s0, alpha1, sb, p1, scut)

        a = N*(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(s_low-s0)**2/(2*sigma**2))
        b = N*p0*np.exp(-alpha*s_high)
        c = N*p1*np.exp(-alpha1*s_high2)

        tmp = np.concatenate((a,b))
        y_out = np.concatenate((tmp,c))
        return y_out  

    
    def LNPLPLfunctionLog(self, s, sigma, alpha, alpha1, sb):
        ''' This function can be used for fitting in Khullar et al. (2021). Returns ln(p(s)).'''
        upper = self.root_finding_upper_lim
        scut = self.s_cut_off
        s0 = FindTheFirstRootPlease(sigma, alpha, alpha1, sb, upper, 1)

        st = s0 + alpha*sigma**2
        if self.debug==True:
            print (sigma, alpha, alpha1, sb, st, s0)

        if (st>sb):
            print ('Skipping this case, st>sb ...')
            return np.ones(len(s))
        if (alpha1>alpha):
            print ('Skipping this case, alpha1>alpha ...')
            return np.ones(len(s))
        
        # Divide the x-range into 3 separate ranges for LN, 1st PL and 2nd PL. 
        s_list = np.where(s<st)[0]
        s_low = np.take(s, s_list)

        s_list = np.where((s>=st)&(s<sb))[0]
        s_high = np.take(s, s_list)

        s_list = np.where(s>=sb)[0]
        s_high2 = np.take(s, s_list)


        p0 = FindPLPLP0(st, sigma, alpha, s0, alpha1, sb)
        p1 = FindPLPLP1(st, sigma, alpha, s0, alpha1, sb)
        #N = FindPLPLN(st, sigma, alpha, p0, s0, alpha1, sb, p1)
        N = FindPLPLNscut(st, sigma, alpha, p0, s0, alpha1, sb, p1, scut)

        a = N*(1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(s_low-s0)**2/(2*sigma**2))
        b = N*p0*np.exp(-alpha*s_high)
        c = N*p1*np.exp(-alpha1*s_high2)

        tmp = np.concatenate((a,b))
        y_out = np.concatenate((tmp,c))

        return np.log(y_out) 


    def LNPLPLfunctionCDF(self, s, sigma, alpha, alpha1, sb):
        ''' This function can be used for fitting. Finds the CDF, P(s) in Khullar et al. (2021). Returns P(s).''' 

        upper = self.root_finding_upper_lim
        scut = self.s_cut_off
        s0 = FindTheFirstRootPlease(sigma, alpha, alpha1, sb, upper, 1)

        st = s0 + alpha*sigma**2
        if self.debug==True:
            print (sigma, alpha, alpha1, sb)
        if (st>sb):
            print ('Skipping this case, st>sb ...')
            return np.ones(len(s))
        if (alpha1>alpha):
            print ('Skipping this case, alpha1>alpha ...')
            return np.ones(len(s))
        
        # Divide the x-range into 3 separate ranges for LN, 1st PL and 2nd PL. 
        s_list = np.where(s<st)[0]
        s_low = np.take(s, s_list)

        s_list = np.where((s>=st)&(s<sb))[0]
        s_high = np.take(s, s_list)

        s_list = np.where(s>=sb)[0]
        s_high2 = np.take(s, s_list)

        p0 = FindPLPLP0(st, sigma, alpha, s0, alpha1, sb)
        p1 = FindPLPLP1(st, sigma, alpha, s0, alpha1, sb)
        #N = FindPLPLN(st, sigma, alpha, p0, s0, alpha1, sb, p1)
        N = FindPLPLNscut(st, sigma, alpha, p0, s0, alpha1, sb, p1, scut)

        a = N/2*scipy.special.erfc((s0-s_low)/(np.sqrt(2)*sigma))

        b = N/2*scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma)) +\
            N*(np.exp(-alpha*st)-np.exp(-alpha*s_high))*p0/alpha

        c = N/2*(2*(np.exp(-alpha*st)-np.exp(-alpha*sb))*p0/alpha +\
                 2*(np.exp(-alpha1*sb)-np.exp(-alpha1*s_high2))*p1/alpha1 +\
                scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma)))


        tmp = np.concatenate((a,b))
        y_out = np.concatenate((tmp,c))

        return y_out  



    def LNPLPLfunctionCDFLog(self, s, sigma, alpha, alpha1, sb):
        ''' This function is used for LN+2PL model fitting in Khullar et al. (2021). 
        See equation 9 in Khullar et al. (2021). This is not the log of the CDF, but rather ln[CDF/(1-CDF)].'''
        if self.debug==True:
            print (sigma, alpha, alpha1, sb)

        upper = self.root_finding_upper_lim
        scut = self.s_cut_off
        s0 = self.FindTheFirstRootPlease(sigma, alpha, alpha1, sb, upper, 1)

        st = s0 + alpha*sigma**2
        if self.debug==True:
            print (sigma, alpha, alpha1, sb, st, s0)
        if (st>sb):
            print ('Skipping this case, st>sb ...')
            return np.ones(len(s))

        if (alpha1>alpha):
            print ('Skipping this case, alpha1>alpha ...')
            return np.ones(len(s))

        # Divide the x-range into 3 separate ranges for LN, 1st PL and 2nd PL. 
        s_list = np.where(s<st)[0]
        s_low = np.take(s, s_list)

        s_list = np.where((s>=st)&(s<sb))[0]
        s_high = np.take(s, s_list)

        s_list = np.where(s>=sb)[0]
        s_high2 = np.take(s, s_list)

        p0 = FindPLPLP0(st, sigma, alpha, s0, alpha1, sb)
        p1 = FindPLPLP1(st, sigma, alpha, s0, alpha1, sb)
        #N = FindPLPLN(st, sigma, alpha, p0, s0, alpha1, sb, p1)
        N = FindPLPLNscut(st, sigma, alpha, p0, s0, alpha1, sb, p1, scut)

        a = N/2*scipy.special.erfc((s0-s_low)/(np.sqrt(2)*sigma))

        b = N/2*scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma)) +\
            N*(np.exp(-alpha*st)-np.exp(-alpha*s_high))*p0/alpha

        c = N/2*(2*(np.exp(-alpha*st)-np.exp(-alpha*sb))*p0/alpha +\
                 2*(np.exp(-alpha1*sb)-np.exp(-alpha1*s_high2))*p1/alpha1 +\
                scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma)))

        tmp = np.concatenate((a,b))
        y_out = np.concatenate((tmp,c))
        if len(np.where(y_out>1.0)[0]):
            return np.ones(len(y_out))

        for i in range (0, len(y_out)):
            y_out[i] = np.log(y_out[i]/(1-y_out[i]))

        return y_out  


    def BootStapper(self, poptPLPL, perrPLPL):
        ''' This is the bootstrapper function. 
        It finds the values of s_0 and subsequently s_t (s_g in Khullar et al. (2021)) by bootstrapping. 

        Here is the procedure for this:
        1. Consider a function f(x_1, x_2, x_3). Generate a set of trial x values by, for each one, picking from a Gaussian \
        centred on x_i with dispersion sigma_i. Let’s call these trial values t_i.
        2. Numerically evaluate f(t_1, t_2, t_3).
        3. Repeat steps 1 and 2 N times; storing the set of values f_n produced in step 2, where n = 1, 2, 3, … N.
        4. The estimate for the central value of f is the average of the f_n’s, and the estimate for the dispersion of f \
        is the dispersion of f_n values.
        '''

        print ('Bootstrapping starts now ...')

        sigma = poptPLPL[0]
        alpha = poptPLPL[1]
        sigma_err = perrPLPL[0]
        alpha_err = perrPLPL[1]
        alpha1 = poptPLPL[2]
        alpha1_err = perrPLPL[2]
        sb = poptPLPL[3]
        sb_err = perrPLPL[3]

        scut = self.s_cut_off
        s0 = optimize.bisect(lambda x: s0rootfuncnew(x, sigma, alpha, alpha1, sb, scut), self.root_finding_lower_lim, \
                             self.root_finding_upper_lim)


        st = s0+alpha*sigma**2
        N = 10000

        sigma_dist = np.random.normal(sigma, sigma_err, N)
        alpha_dist = np.random.normal(alpha, alpha_err, N)
        alpha1_dist = np.random.normal(alpha1, alpha1_err, N)
        sb_dist = np.random.normal(sb, sb_err, N)
        s0_list = []
        for i in range(0, N):
            try:
                val = optimize.bisect(lambda x: s0rootfuncnew(x, sigma_dist[i], \
                                                alpha_dist[i], alpha1_dist[i], sb_dist[i], scut), \
                                      self.root_finding_lower_lim, self.root_finding_upper_lim)
                s0_list.append(val)
            except ValueError:
                continue

        s0_list = np.array(s0_list)
        s0bootmean = s0_list.mean()
        s0bootstd = np.std(s0_list)

        if self.debug==True:
            print ('First attempt values -', s0, s0bootmean, s0bootstd)

        ###################################
        ###################################
        ## if the difference between bootstrap and no bootstrap is more than 0.5, repeat with higher N
        if abs(s0-s0bootmean)>=0.5:
            if self.debug==True:
                print ('Second attempt...')

            s0bootmean_old = s0_list.mean()
            s0bootstd_old = np.std(s0_list)

            N = 50000
            sigma_dist = np.random.normal(sigma, sigma_err, N)
            alpha_dist = np.random.normal(alpha, alpha_err, N)
            alpha1_dist = np.random.normal(alpha1, alpha1_err, N)
            sb_dist = np.random.normal(sb, sb_err, N)
            s0_list = []
            for i in range(0, N):
                try:
                    val = optimize.bisect(lambda x: s0rootfuncnew(x, sigma_dist[i], \
                                                    alpha_dist[i], alpha1_dist[i], sb_dist[i], scut), -10, 1)
                    s0_list.append(val)
                except ValueError:
                    continue

            s0_list = np.array(s0_list)
            s0bootmean = s0_list.mean()
            s0bootstd = np.std(s0_list)

            if self.debug==True:
                print ('Second attempt values -', s0, s0bootmean, s0bootstd)

            #### Check if problem with bootstrap, check for convergence
            if abs(s0bootmean_old-s0bootmean)>=0.1:
                #if abs(s0bootmean_old-s0bootmean)>=0.5
                if debug==True:
                    print ('Third attempt...Bootstrap not converged')
                N = 100000
                sigma_dist = np.random.normal(sigma, sigma_err, N)
                alpha_dist = np.random.normal(alpha, alpha_err, N)
                alpha1_dist = np.random.normal(alpha1, alpha1_err, N)
                sb_dist = np.random.normal(sb, sb_err, N)
                s0_list = []
                for i in range(0, N):
                    try:
                        val = optimize.bisect(lambda x: s0rootfuncnew(x, sigma_dist[i], \
                                                        alpha_dist[i], alpha1_dist[i], sb_dist[i], scut), -10, 1)
                        s0_list.append(val)
                    except ValueError:
                        continue

                s0_list = np.array(s0_list)
                s0bootmean = s0_list.mean()
                s0bootstd = np.std(s0_list)


        print ('Bootstrapping results - ',s0, s0bootmean, s0bootstd)

        ###################################
        ###################################
        #### Return the bootstrapped values
        s0_err = s0bootstd
        st_err = FindstError(s0bootmean, alpha, sigma, s0_err, alpha_err, sigma_err)
        st = s0bootmean+alpha*sigma**2
        if abs(s0-s0bootmean)>=0.5:
            print ('Bootstrapping not effective? Weird case...')
            return s0bootmean, s0_err, st, st_err


        return s0bootmean, s0_err, st, st_err