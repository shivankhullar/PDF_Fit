import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import curve_fit

# ==================================================================== #
# ==================================================================== #
# ================ For the Lognormal + Power-Law model =============== #
# ==================================================================== #
# ==================================================================== #

### Note that we use a different convention here. 
### The variables alpha, st, alpha_1, s_b are alpha_g, s_g, alpha_d and s_d in Khullar et al. (2021) respectively. 

## Find the normalization constant
def FindN(st, sigma_s, alpha, p0, s0):
    ''' This function returns the value of N, the normalization constant. See equation A1 in Khullar et al. (2021).'''
    return ((p0/alpha)*np.exp(-alpha*st)+0.5*(scipy.special.erfc((s0-st)/(2**0.5*sigma_s))))**(-1)

## Find the quantity p0
def FindP0(st, sigma_s, alpha, s0):
    ''' This function returns the value of p_0. See equation A3 in Khullar et al. (2021). '''
    return (st-s0)*np.exp(alpha*st-(st-s0)**2/(2*sigma_s**2))/(alpha*np.sqrt(2*np.pi)*sigma_s**3)


## Find the mean
def Finds0(sigma_s, alpha):
    ''' This function returns the value of the mean of the LN+PL distribution, s_0. 
    See equation A5 in Khullar et al. (2021). '''
    numerator = (-1+alpha)*(np.sqrt(2) + alpha*np.exp(alpha**2*sigma_s**2/2)*np.sqrt(np.pi)*sigma_s*(1+scipy.special.erf(alpha*sigma_s/2)))
    denominator1 = np.sqrt(2)*alpha*np.exp(alpha*sigma_s**2)
    denominator2 = (-1+alpha)*alpha*np.exp((1+alpha**2)*sigma_s**2/2)*np.sqrt(np.pi)*sigma_s*(1+scipy.special.erf((-1+alpha)*sigma_s/2))
    denominator = denominator1+denominator2
    return np.log((numerator)/(denominator))


### Function for finding out errors in s0, so this gives \partial(s0)/\partial(sigma)
def FindDels0Delsigma(sigma, alpha):
    ''' This function returns the partial derivative of s_0 w.r.t sigma or sigma_s.
    Used ultimately in finding the error in s_0. '''
    term1 = -1/sigma
    term2 = -sigma
    term3 = 1/(sigma + (-1+alpha)*np.exp(0.5*(-1+alpha)**2*sigma**2)*np.sqrt(np.pi/2)*sigma**2*\
              (1+scipy.special.erf((-1+alpha)*sigma/np.sqrt(2))))
    term4 = 1/(sigma + (np.sqrt(2/np.pi)*np.exp(-0.5*alpha**2*sigma**2))/(alpha+alpha*\
                                                                          scipy.special.erf(alpha*sigma/np.sqrt(2))))
    full = term1 + term2 + term3 + term4
    
    return full

### Function for finding out errors in s0, so this gives \partial(s0)/\partial(alpha)
def FindDels0Delalpha(sigma, alpha):
    ''' This function returns the partial derivative of s_0 w.r.t alpha.
    Used ultimately in finding the error in s_0. '''
    numterm1 = 2*np.exp(alpha*sigma**2) 
    numterm2 = alpha**2*np.exp(alpha*(2+alpha)*sigma**2/2)*np.sqrt(2*np.pi)*sigma*\
    (1+scipy.special.erf(alpha*sigma/np.sqrt(2)))
    numterm3 = (-1+alpha)**2*np.exp((1+alpha)**2*sigma**2/2)*np.sqrt(2*np.pi)*sigma*\
    (-2+scipy.special.erfc((-1+alpha)*sigma/np.sqrt(2)))
    numerator = numterm1+numterm2+numterm3
    a = (-1+alpha)*alpha
    b = (np.sqrt(2)*np.exp(alpha*sigma**2) + (-1+alpha)*np.exp((1+alpha**2)*sigma**2/2)*np.sqrt(np.pi)*\
        sigma*(1+scipy.special.erf((-1+alpha)*sigma/np.sqrt(2))))
    c = np.sqrt(2)+alpha*np.exp(alpha**2*sigma**2/2)*np.sqrt(np.pi)*sigma*(1+scipy.special.erf(alpha*sigma/np.sqrt(2)))
    denominator = a*b*c
    full = numerator/denominator
    return full

### Finally find error in s0
def Finds0err(sigma, sigma_err, alpha, alpha_err):
    ''' This function returns the error in s_0. '''
    del_s0_by_del_sigma = FindDels0Delsigma(sigma, alpha)
    del_s0_by_del_alpha = FindDels0Delalpha(sigma, alpha)
    del_s0 = np.sqrt((del_s0_by_del_sigma*sigma_err)**2+(del_s0_by_del_alpha*alpha_err)**2)
    return del_s0




## The PDF function LN+PL. Only sigma and alpha are the free parameters
def LNPLfunction(s, sigma_s, alpha):
    ''' This is the function that describes the LNPL function. See equation 7 in Khullar et al. (2021). 
    Returns p(s). '''
    s0 = Finds0(sigma_s, alpha)
    st = s0 + alpha*sigma_s**2
    
    ### Splitting the x array into LN and PL part called s_low and s_high
    s_list = np.where(s<st)[0]
    s_low = np.take(s, s_list)

    s_list = np.where(s>=st)[0]
    s_high = np.take(s, s_list)
        
    p0 = FindP0(st, sigma_s, alpha, s0)
    N = FindN(st, sigma_s, alpha, p0, s0)
    
    y_out = np.concatenate((N*(1/(np.sqrt(2*np.pi)*sigma_s))*np.exp(-(s_low-s0)**2/(2*sigma_s**2)),\
                            N*p0*np.exp(-alpha*s_high))) 
    return y_out          
        
    
    
#### Function for the CDF instead of PDF. 
def LNPLfunctionCDF(s, sigma_s, alpha):
    ''' This function can be used for fitting. Finds the CDF, P(s) in Khullar et al. (2021). Returns P(s).''' 
    
    s0 = Finds0(sigma_s, alpha)
    st = s0 + alpha*sigma_s**2
    
    ### Splitting the x array into LN and PL part called s_low and s_high
    s_list = np.where(s<st)[0]
    s_low = np.take(s, s_list)

    s_list = np.where(s>=st)[0]
    s_high = np.take(s, s_list)
    
    p0 = FindP0(st, sigma_s, alpha, s0)
    N = FindN(st, sigma_s, alpha, p0, s0)
    a = N/2*scipy.special.erfc(-(s_low-s0)/(np.sqrt(2)*sigma_s))
    
    c = N*(p0/alpha)*(np.exp(-alpha*st)) + N/2*scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma_s))
    b = N*(p0/alpha)*(-1*np.exp(-alpha*s_high)) + c
    
    y_out = np.concatenate((a, b))
    
    return y_out  


#### Function for y = log(CDF/(1-CDF)) -- since this is what we use to fit so that there's more weight to the tail
def LNPLfunctionCDFLog(s, sigma_s, alpha):  
    ''' This function is used for LN+PL fitting in Khullar et al. (2021). See equation 9 in Khullar et al. (2021). '''
    s0 = Finds0(sigma_s, alpha)
    st = s0 + alpha*sigma_s**2
    
    ### Splitting the x array into LN and PL part called s_low and s_high
    s_list = np.where(s<st)[0]
    s_low = np.take(s, s_list)

    s_list = np.where(s>=st)[0]
    s_high = np.take(s, s_list)
    
    s_high = np.array(s_high)
    p0 = FindP0(st, sigma_s, alpha, s0)
    N = FindN(st, sigma_s, alpha, p0, s0)
    a = N/2*scipy.special.erfc(-(s_low-s0)/(np.sqrt(2)*sigma_s))
    
    c = N*(p0/alpha)*(np.exp(-alpha*st)) + N/2*scipy.special.erfc((s0-st)/(np.sqrt(2)*sigma_s))
    b = N*(p0/alpha)*(-1*np.exp(-alpha*s_high)) + c
    
    y_out = np.concatenate((a, b))
    for i in range (0, len(y_out)):
        y_out[i] = np.log(y_out[i]/(1-y_out[i]))
    
    return y_out  


