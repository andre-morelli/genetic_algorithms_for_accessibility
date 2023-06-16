import numpy as np

def cumulative(x, t):
    """
    Calculate cumulative decay for array of observations.
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    t : float
        Cost cap.
    
    Returns
    -------
    Numpy array
    """
    return (x<=t)*1

def inv_exp(x,b):
    """
    Calculate inverse exponential decay for array of observations.
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    b : float
        calibration parameter.
    
    Returns
    -------
    Numpy array
    """
    return np.exp(-b*x)

def cumulative_linear(x,t):
    """
    Calculate linear cumulative decay for array of observations.
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    t : float
        Point where weight is zero.
    
    Returns
    -------
    Numpy array
    """
    return (x<=t)*(1-x/t)

def cumulative_gauss(x,t,v):
    """
    Calculate cumulative gaussian decay for array of observations.
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    t : float
        Cost cap.
    v : float
        Calibration parameter. Rate of decay after cost cap
    
    Returns
    -------
    Numpy array
    """
    return (x<=t)*1 + (x>t)*np.exp(-(x-t)/v)

def mod_gauss(x,b):
    """
    Calculate modified gaussian decay for array of observations.
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    b : float
        calibration parameter.
    
    Returns
    -------
    Numpy array
    """
    return np.exp(-x**2/b)

def soft_threshold(x, t=500, k=5):
    """
    Calculate soft threshold decay funciton for array of observations.
    source: Higgs, C., Badland, H., Simons, K. et al. The Urban Liveability 
    Index: developing a policy-relevant urban liveability composite measure 
    and evaluating associations with transport mode choice. Int J Health 
    Geogr 18, 14 (2019). https://doi.org/10.1186/s12942-019-0178-8
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    t : float
        function calibration parameter (point at which access score = 0.5).
    k : float
        function calibration parameter
    
    Returns
    -------
    Numpy array
    """
    return (1+np.exp(k*(x-t)/t))**-1

def mod_log_logit(x,k,c):
    """
    Calculate modified log-logit decay for array of observations.
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    k : float
        calibration parameter.
    c : float
        calibration parameter.
    
    Returns
    -------
    Numpy array
    """
    return np.exp(-b*x)
    return 1/(1+np.exp(k+c*np.log(x+(x==0)*1e-8)))

def inv_pow(x,b):
    """
    Calculate inverse power decay for array of observations.
    
    Parameters
    ----------
    x : numpy array
        Array where every entry is a travel cost.
    b : float
        exponent.
    
    Returns
    -------
    Numpy array
    """
    x = x + (x==0)*1e-8
    return (x<=1)*1 + (x>1)*(x**-b)