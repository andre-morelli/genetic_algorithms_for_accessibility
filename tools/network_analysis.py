import osmnx as ox
import numpy as np

def get_orientation_entropy(G, weight=None, n=36):
    """
    Calculate orientation entropy for a given graph.
    
    source: Boeing, G. Urban spatial order: street network orientation, 
    configuration, and entropy. Appl Netw Sci 4, 67 (2019). 
    https://doi.org/10.1007/s41109-019-0189-1 

    Parameters
    ----------
    G : NetworkX MultiDiGraph 
        Network's graph.
    weight: string
        Edge attribute to weight by. If None, each edge has same weight.
    n: int 
        Number of bins. Divides the full circle (360°) into bins
   
    Returns
    -------
    float
    """
    
    Gu = ox.add_edge_bearings(ox.get_undirected(G))

    if weight != None:
        # weight bearings by NUMERIC attribute
        city_bearings = []
        for u, v, k, d in Gu.edges(keys=True, data=True):
            city_bearings.extend([d['bearing']] * int(d[weight]))
        b = pd.Series(city_bearings)
        bearings = pd.concat([b, b.map(_reverse_bearing)]).reset_index(drop='True')
    else:
        # don't weight bearings, just take one value per street segment
        b = pd.Series([d['bearing'] for u, v, k, d in Gu.edges(keys=True, data=True)])
        bearings = pd.concat([b, b.map(_reverse_bearing)]).reset_index(drop='True')
    
    n = n * 2
    bins = np.arange(n + 1) * 360 / n
    counts, _ = np.histogram(bearings, bins=bins)
    
    # move the last bin to front to complete circle
    counts = np.roll(counts, 1)
    counts = counts[::2] + counts[1::2]
    
    probs = counts/counts.sum()
    return (-np.log(probs)*probs).sum()
    
    
def get_float_entropy(data, weight=None, n=36, vmin=None, vmax=None):
    """
    Calculate entropy for a given dataset.
    
    Parameters
    ----------
    data : unidimensinal array-like or None
        data to calculate entropy on.
    weight: unidimensinal array-like (only ints)
        Values to weight data. Must have same size as data or None. If None, each edge has same weight.
    n: int 
        Number of bins. Divides data into equal sized bins
    vmin: float
        lower cap of bins
    vmax: float
        upper cap of bins
    Returns
    -------
    float
    """
    assert vmin==vmax==None or vmin<vmax, "vmin must be smaller than vmax"
    assert type(n)==int and n>=2, "number of bins must be an integer bigger than 1"
    assert weight == None or len(weigth)==len(data), "if weights != None, length of weights must be the same as data"
    if vmin==None:
        vmin = min(data)
    if vmax==None:
        vmax=max(data)
    if weight != None:
        # weight bearings by NUMERIC attribute
        d = []
        for value,w in zip(data,weight):
            d+=[value] * int(w)
        data = np.array(d)
    
    counts, _ = np.histogram(data,range=(vmin,vmax), bins=n)
    
    probs = counts/counts.sum()
    probs = np.array([k for k in probs if k>0]) #ignore probability zero (if p==0, p*log(p)->0)
    return (-np.log(probs)*probs).sum()
    
def gini(x, weights=None):
    """
    Calculate gini coefficient for a given array.

    Parameters
    ----------
    x : numpy unidimensional array or array-like data (list, tuple, etc.)
        Data array.
    w : numpy unidimensional array or array-like data (list, tuple, etc.)
        weights for data. Must be same size as x.
   
    Returns
    -------
    float
    """
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if weights is not None:
        weights = np.asarray(weights)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = weights[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def concentration(values, upper_strata=1., weights=None):
    """
    Calculate concentration coefficient for a given array. This coefficient
    shows the proportion of the total value attributed to the upper strata

    Parameters
    ----------
    values : list-like
        list of values.
    upper_strata : float
        upper percentage of values to consider. By default, the function
        returns the concentration on the 1% strata with biggest associated
        values
   
    Returns
    -------
    float
    """
    
    # upper_strata in percentage
    if weights != None:
        array = list(zip(values,weights))
        array = list(sorted(array, reverse=True))
    else:
        weights = [1]*len(values)
        array = list(zip(values,weights))
        array = list(sorted(array, reverse=True))
    values = np.array(values)
    weights = np.array(weights)
    weight_total = weights.sum()
    upper_concentration = 0
    counter = 0
    for value, w in array:
        if counter >= weight_total*upper_strata/100:
            break
        counter += w
        upper_concentration += value*w
    return (upper_concentration/(weights*values).sum())
    

def get_attr_gini_coef(G, attribute, weight=None, kind = 'edge'):
    """
    Calculate gini coefficient for data in a NetworkX graph.

    Parameters
    ----------
    G : NetworkX Graph structure
        Graph of the network.
    attribute: string
        Graph attribute to calculate gini on
    weight : string
        Graph attribute to weight by.
    kind: 'edge' or 'node'
        Attribute belongs to edges or nodes
   
    Returns
    -------
    float
    """
    
    if kind == 'edge':
        attrs = [attr for n1, n2, attr in G.edges(data=attribute)]
        if weight != None:
            weight = [int(attr) for n1, n2, attr in G.edges(data=weight)]
    elif kind == 'node':
        attrs = [attr for n, attr in G.nodes(data=attribute)]
        if weight != None:
            weight = [int(attr) for n, attr in G.nodes(data=weight)]
    return gini(attrs, weight)

def get_attr_concentration_coef(G, attribute, upper_strata=1, weight=None, kind = 'edge'):
    """
    Calculate concentration coefficient for data in a NetworkX graph.

    Parameters
    ----------
    G : NetworkX Graph structure
        Graph of the network.
    attribute: string
        Graph attribute to calculate gini on
    upper_strata : float
        upper percentage of values to consider. By default, the function
        returns the concentration on the 1% strata with biggest associated
        values
    weight : string
        Graph attribute to weight by.
    kind: 'edge' or 'node'
        Attribute belongs to edges or nodes
   
    Returns
    -------
    float
    """
    
    if kind == 'edge':
        attrs = [attr for n1, n2, attr in G.edges(data=attribute)]
        if weight != None:
            weight = [int(attr) for n1, n2, attr in G.edges(data=weight)]
    elif kind == 'node':
        attrs = [attr for n, attr in G.nodes(data=attribute)]
        if weight != None:
            weight = [int(attr) for n, attr in G.nodes(data=weight)]
    return concentration(attrs, weights = weight, upper_strata=upper_strata)