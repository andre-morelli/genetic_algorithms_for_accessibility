from .utils import get_igraph
from .decay_funcs import *

import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd

from math import ceil
import threading
from joblib import Parallel, delayed
import itertools

def get_reference_nodes(gdf, G, k=5, seed=None, return_ref_values=False,
                        ref_val_column=None):
    """
    Define which nodes will be attributed to each traffic zone in shrotest-paths algorithms.

    Parameters
    ----------
    gdf: Geopandas GeoDataFrame (areas)
        GeoDataFrame containing the traffic zones.
    G : NetworkX DiGraph or Graph 
        Network's graphs. Ideally, crs must be the same as gdf
    k:  int
        Number of nodes per zone. Each zone will have k or less ref nodes. 
    seed: 
        Random state of the process. Set this for comparizon analysis.
    return_ref_values: boolean
        If True, return the values associated with each node (population or opportunities).
    ref_val_column: string
        column of gdf containing values to distribute among nodes.

    Returns
    -------
    numpy array, sequence of nodes; dict, nodes per zone; (optionally) numpy array, reference 
    values per node.
    """
    
    nodes = ox.graph_to_gdfs(G, edges=False)
    joined_gdf = gpd.sjoin(gdf, nodes.to_crs(gdf.crs))
    
    targets = []
    z_nodes = {}

    if return_ref_values:
        # for one attribute
        if type(ref_val_column) == str:
            vals = []
            
        elif isinstance(ref_val_column, (list, tuple, np.array)):
            vals = {}
            
        else:
            raise ValueError('ref_val_column must be string or list of strings')
            
    for i, row in gdf.iterrows():
        if i not in joined_gdf.index:
            raise ValueError(f'No node within zone {i}')
        temp = joined_gdf.loc()[[i]]
        
        if len(temp)<=k:
            
            ns = list(temp['index_right'])
            if return_ref_values:
                # maybe organize this.
                if type(vals) == list:
                    vals += [row[ref_val_column]/len(ns)]*len(ns)
                
                else:
                    for ref in ref_val_column:
                        if ref not in vals:
                            vals[ref] = []
                        vals[ref] += [row[ref]/len(ns)]*len(ns)
            targets += ns
            # add referencing to know what nodes belong to which zone
            z_nodes[i] = ns
        else:
            ns = list(temp['index_right'].sample(k, random_state=seed))
            
            #getting reference values
            if return_ref_values:
                if type(vals) == list:
                    vals += [row[ref_val_column]/len(ns)]*len(ns)
                else:
                    for ref in ref_val_column:
                        if ref not in vals:
                            vals[ref] = []
                        vals[ref] += [row[ref]/len(ns)]*len(ns)
                        
            targets += ns
            z_nodes[i] = ns
    if return_ref_values:
        return targets, z_nodes, vals
    else:
        return targets, z_nodes

def get_costs(gdf, G, weight='length', seed=None, 
                    round_trip=False, targets = None, z_nodes = None,
                    k = 5, return_as_matrix=True):
    """
    Get the costs between pairs of zones as an array or matrix.

    Parameters
    ----------
    gdf: Geopandas GeoDataFrame (areas)
        GeoDataFrame containing the traffic zones.
    G : NetworkX DiGraph or Graph 
        Network's graphs. Ideally, crs must be the same as gdf
    weight: string
        Edge attribute to weight shortest paths by.
    seed: 
        Random state of the process. Set this for comparizon analysis.
    round_trip: boolean
        If True, the distance between two points will be the average between the incoming
        and the outgoing paths
    targets: numpy array
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where cost has to be calculated multiple times.
    z_nodes: dictionary or mapping
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where cost has to be calculated multiple times.
    k:  int
        Number of nodes per zone. Each zone will have k or less ref nodes.
    return_as_matrix: boolean
        If True, the function returns a cost matrix (zone to zone cost), but is slower.

    Returns
    -------
    numpy array.
    """
    
    Gig = get_igraph(G, [weight])
    node_dict = {}
    for node in Gig.vs:
        node_dict[node['name']] = node.index
    
    if targets is None:
        #this take a looong time
        targets, z_nodes = get_reference_nodes(gdf, G, k=k, seed=seed)
    elif z_nodes is None:
        raise ValueError('if targets is provided, the nodes by zone dictionary' +
                         '(z_nodes) must be provided in the form' + 
                         '{zone_id: list_of_reference_nodes}')
    
    nig = [node_dict[n] for n in targets]
    dists = Gig.shortest_paths_dijkstra(nig,nig,weights=weight,
                                            mode='out')
    dists = np.array(dists)
    
    if round_trip:
        dists = dists + np.array(Gig.shortest_paths_dijkstra(nig,nig,
                                                             weights=weight,
                                                             mode='in'))
        dists = dists*.5
    targets = np.array(targets)
    indices = {i : np.where( np.isin(targets, z_nodes[i]) ) for i in gdf.index}
    costs = {}
    
    for i in gdf.index:
        i1 = indices[i]
        
        if return_as_matrix:
            #this is slow
            costs[i] = {}
            for j in gdf.index:
                i2 = indices[j]
                cs = dists[i1][:, i2].flatten()
                if len(cs[~np.isinf(cs)])>0:
                    costs[i][j] = np.nanmean(cs[~np.isinf(cs)])
                else:
                    costs[i][j] = np.inf
        else:
            costs[i] = np.nanmean(dists[i1], axis=0)
    return costs

def accessibility(gdf, G, weight='length', round_trip=False,
                 k=5, seed=None, opportunities_column=None,
                 population_column=None, competition=False,
                 decay_func = cumulative, decay_kws={'t':1800},
                 targets=None, z_nodes=None, vals=None):
    """
    Get the costs between pairs of zones as an array or matrix.

    Parameters
    ----------
    gdf: Geopandas GeoDataFrame (areas)
        GeoDataFrame containing the traffic zones.
    G : NetworkX DiGraph or Graph 
        Network's graphs. Ideally, crs must be the same as gdf
    weight: string
        Edge attribute to weight shortest paths by.
    round_trip: boolean
        If True, the distance between two points will be the average between the incoming
        and the outgoing paths
    k:  int
        Number of nodes per zone. Ignored ir targets is provided.
    seed: 
        Random state of the process. Ignored ir targets is provided.
    opportunities_column: string
        Column that provides opportunity number in gdf.
    population_column: string
        Column that provides population number in gdf.
    competition: boolean
        If True, calculates accessibility through a two step flaoting catchment area (2SFCA) 
        process
    decay_func: function
        Distance decay function
    decay_kws: dictionary
        parameters for the distance decay function
    targets: numpy array
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    z_nodes: dictionary or mapping
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    vals: numpy array
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.

    Returns
    -------
    dictionary
    """
    
    ref_cols = [opportunities_column, population_column]
    if targets is None:
        targets, z_nodes, vals = get_reference_nodes(gdf, G, k=k, seed=seed, 
                                                     return_ref_values=True,
                                                     ref_val_column=ref_cols)
    opportunities = np.array(vals[opportunities_column])
    population = np.array(vals[population_column])
    
     #get costs
    c = get_costs(gdf, G, targets=targets, return_as_matrix=False,
                        z_nodes=z_nodes, weight=weight, round_trip=round_trip)
    if competition:
        # calculate first order catchment
        ctch = {}
        for j in c:
            ctch[j] = (decay_func(c[j], **decay_kws)*population).sum()
        # calculate accessibilities dividing by catchment
        acc = {}
        for j in c:
            # create a list of the same length of the values
            corr = []
            for z, l in z_nodes.items():
                corr += [ctch[z]]*len(l)
            assert len(corr) == len(opportunities), 'Something went wrong'
            #correct values (and make sure cr is never zero)
            corrected_opps = [o/(cr+.1) for o,cr in zip(opportunities, corr)]
            acc[j] = (decay_func(c[j], **decay_kws)*corrected_opps).sum()
    else:
        # calculate first order catchment
        acc = {}
        for j in c:
            acc[j] = (decay_func(c[j], **decay_kws)*opportunities).sum()
    return acc


# Accessibility network metrics

def _get_edges(e_seqs,Gig,weight='length'):
    """
    Returns the sequence of edges in the original graph from an iGraph version.

    Parameters
    ----------
    e_seqs: list like
        sequence of edges resulting from igraph's get_shortest_paths in epath mode.
    Gig : igraphs's Graph 
        Network's graphs.
    weight: string
        Edge attribute shortest paths were weighted by.
    
    Returns
    -------
    list of names, list of weights
    """
    r = []
    s = []
    for seq in e_seqs:
        if len(seq)<1:
            r.append([])
            s.append(0)
        else:
            r.append(Gig.es[seq]['orig_name'])
            s.append(sum(Gig.es[seq][weight]))
    return r,s

def _update_elements(el_mat, path_seq, weights, 
                     z1, time, opportunities, population, 
                     dist_decay, w_bottom=1e-4):
    """
    updates elements in a path statistiscs list.

    Parameters
    ----------
    el_mat: list
        Path statistiscs list
    path_seq: list
        Sequence of paths.
    weight: list
        Weights for each path.
    z1: list
        Zone where paths initiate.
    time: list
        Cost of each path
    opportunities: list
        Number of opportunities where the paths terminate.
    population: list
        Number of persons where the paths initiate.
    dist_decay: list
        Value of distance decay function for each path
    w_bottom: float
        Ignore all paths with weight smaller or equal to this.
        Higher values cut down on memory requirements
    
    Returns
    -------
    list
    """
    if weights is None: 
        weights=[1]*len(e_seq)
    else:
        weights=[n if n==n else 0 for n in weights]
    el_mat+=[(w,z,t,p,o,pp,dd) 
             for w,z,t,p,o,pp,dd in 
             zip(weights, z1, time, path_seq,
                 opportunities, population, 
                 dist_decay) 
             if w>w_bottom] #this also filter out nans
    
    return el_mat

def _calculate_paths_stats(sources, gdf, Gig, node_dict, z_nodes,
                          nig, weight, decay_func, indices,
                          decay_kws, opportunities, corr,
                          population):
    """
    Returns statistics of a set of shortest paths. Used for parallel computation.

    Parameters
    ----------
    sources: list like
        sequence of source nodes to perform shortest paths.
    gdf: Geopandas GeoDataFrame (areas)
        GeoDataFrame containing the traffic zones.
    Gig : igraphs's Graph 
        Network's graphs.
    node_dict: dictionary or mapping
        maps original nodes to nodes in igraph version
    z_nodes: dictionary or mapping
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    nig: list like 
        sequence of target nodes to perform shortest paths
    weight: string
        Edge attribute to weight shortest paths by.
    decay_func: function
        Distance decay function
    indices: dictionary or mapping
        indices refering to nodes by zone
    decay_kws: dictionary
        Distance decay funciton parameters
    opportunities: array
        Opportunities per node in nig.
    corr: array
        Correction factors (for 2SFCA)
    population: array
        Population per node in nig.
    
    Returns
    -------
    numpy list.
    """
    sts= []
    for i in sources:
        for source in [node_dict[n] for n in z_nodes[i]]:
            e_seq = Gig.get_shortest_paths(source, nig, weights=weight, output='epath')
            e_seq,ts = _get_edges(e_seq, Gig, weight=weight)
            dist_decay = decay_func(np.array(ts), **decay_kws)
            w = (dist_decay * 
                 opportunities * 
                 population[indices[i]][0] / 
                 corr)
            sts.append(dict(path_seq=e_seq, weights=w, z1=[i]*len(w), 
                           time=ts, opportunities=opportunities,
                           population=population, 
                           dist_decay=dist_decay))
    return sts

def get_path_db(gdf, G, targets, z_nodes, vals, weight='length', workers=1,
                round_trip=False, k = 5, decay_func = cumulative, 
                decay_kws={'t':1800}, opportunities_column = None, 
                population_column = None, seed=None, competition=False):
    """
    Get the statistics of shortest paths between zones.

    Parameters
    ----------
    gdf: Geopandas GeoDataFrame (areas)
        GeoDataFrame containing the traffic zones.
    G : NetworkX DiGraph or Graph 
        Network's graphs. Ideally, crs must be the same as gdf
    targets: numpy array
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    z_nodes: dictionary or mapping
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    vals: numpy array
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    weight: string
        Edge attribute to weight shortest paths by.
    workers: int
        number of threads to use in computation
    round_trip: boolean
        If True, the distance between two points will be the average between the incoming
        and the outgoing paths
    k:  int
        Number of nodes per zone. Ignored ir targets is provided.
    decay_func: function
        Distance decay function
    decay_kws: dictionary
        parameters for the distance decay function
    opportunities_column: string
        Column that provides opportunity number in gdf.
    population_column: string
        Column that provides population number in gdf.
    seed: 
        Random state of the process. Ignored ir targets is provided.
    competition: boolean
        If True, calculates accessibility through a two step flaoting catchment area (2SFCA) 
        process
    
    Returns
    -------
    list
    """
    # turn graph into iGraph class for faster computation
    Gig = get_igraph(G, [weight])
    Gig.es['orig_name'] = list(G.edges)
    node_dict = {}
    for node in Gig.vs:
        node_dict[node['name']] = node.index
    
    ref_cols = [opportunities_column, population_column]
    # get opportunities and population in array format
    opportunities = np.array(vals[opportunities_column])
    population = np.array(vals[population_column])
    
    nig = [node_dict[n] for n in targets]
    
    if competition:
        # calculate correction factors
        ctch = accessibility(zs, G, opportunities_column=opportunities_column, 
                             population_column = population_column, competition=False,
                             weight=weight, decay_func=decay_func, decay_kws=decay_kws,
                             targets=targets, z_nodes=z_nodes, vals=vals)
        corr = {}
        for i,j in z_nodes.items():
            corr += [ ctch[i] ] * len(j)
        corr = np.array(corr)
    else:
        # all correction factors are set to 1
        corr = np.array([1]*len(targets))
        
    indices = {i : np.where( np.isin(targets, z_nodes[i]) ) for i in gdf.index}
    
    # expansion to better take advantage of logical threads
    if workers == -1:
        processes = threading.active_count()*2
    else:
        processes = workers*2
    assert (workers > 0 or workers==-1), 'Workers must be positive integer or -1 for all threads'
    
    # divide the DB to handle parallelization
    section_length = ceil(len(gdf)/processes)
    work_loads = [gdf.index[n*section_length: n*section_length + section_length] 
                  for n in range(processes)]
    # run _calculate_path_stats in parallel
    sts = Parallel(n_jobs=workers)(delayed(_calculate_paths_stats)(sources, gdf, Gig, node_dict, z_nodes,
                                                                  nig, weight, decay_func, indices,
                                                                  decay_kws, opportunities, corr,
                                                                  population) 
                                              for sources in work_loads)
    # flatten the array to create info array
    sts = itertools.chain(*sts)
    # initialize info array
    mat = []
    for row in sts:
        mat = _update_elements(mat, **row)
    return mat

def betweenness_accessibility(gdf, G, weight='length', round_trip=False,
                             k=5, seed=None, opportunities_column=None,
                             population_column=None, competition=False,
                             decay_func = cumulative, decay_kws={'t':1800},
                             targets=None, z_nodes=None, vals=None, workers = 1):
    """
    Calculates betweenness accessibility. 
    Reference: Sarlas, Georgios, Antonio Páez, and Kay W. Axhausen. 
    "Betweenness-accessibility: Estimating impacts of accessibility on networks." 
    Journal of Transport Geography 84 (2020): 102680.

    Parameters
    ----------
    gdf: Geopandas GeoDataFrame (areas)
        GeoDataFrame containing the traffic zones.
    G : NetworkX DiGraph or Graph 
        Network's graphs. Ideally, crs must be the same as gdf
    weight: string
        Edge attribute to weight shortest paths by.
    round_trip: boolean
        If True, the distance between two points will be the average between the incoming
        and the outgoing paths
    k:  int
        Number of nodes per zone. Ignored ir targets is provided.
    seed: 
        Random state of the process. Ignored ir targets is provided.
    opportunities_column: string
        Column that provides opportunity number in gdf.
    population_column: string
        Column that provides population number in gdf.
    competition: boolean
        If True, calculates accessibility through a two step flaoting catchment area (2SFCA) 
        process
    decay_func: function
        Distance decay function
    decay_kws: dictionary
        parameters for the distance decay function
    targets: numpy array
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    z_nodes: dictionary or mapping
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    vals: numpy array
        Optional. Result from get_reference_nodes. If provided, the calculation is faster. 
        Recomended for processes where accessibility has to be calculated multiple times.
    workers: int
        number of threads to use in computation
    
    Returns
    -------
    list
    """
    
    ref_cols = [opportunities_column, population_column]
    if targets is None:
        targets, z_nodes, vals = get_reference_nodes(gdf, G, k=k, seed=seed, 
                                                     return_ref_values=True,
                                                     ref_val_column=ref_cols)
    opportunities = np.array(vals[opportunities_column])
    population = np.array(vals[population_column])
    
    mat = get_path_db(gdf, G, weight=weight, seed=seed, vals=vals,
                     round_trip=round_trip, targets=targets, z_nodes=z_nodes,
                     k=k, decay_func=decay_func, decay_kws=decay_kws,
                     opportunities_column=opportunities_column, workers=workers,
                     population_column=population_column, competition=competition)
    
    return mat

def add_edge_loads(G, path_mat, filter_areas=None,
                   name='load', inplace=True,
                   return_cost_hist=False):
    """
    Adds loads into edges of a graph from an egde statistics list.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        Network's graphs.
    filter_areas: list like
        If provided, only consider paths originating in one of the zones in this list.
    name: string
        string to name the attribute in the graph
    inplace:  boolean
        If True, modifies the original graph.
    return_cost_hist: 
       If True, returns a histogram of the path's costs
    
    Returns
    -------
    NetworkX Graph and/or list
    """
    
    if not inplace:
        G = G.copy()
    loads = {}.fromkeys(G.edges,0)
    tot_time = {}.fromkeys(G.edges,0)
    tot_routes = {}.fromkeys(G.edges,0)
    if return_cost_hist:
        cost_hist = {}.fromkeys(G.edges)
        for e in cost_hist:
            cost_hist[e] = []
    for w,z,t,path,opp,pop,dist_decay in path_mat:
        if filter_areas is None or z in filter_areas:
            for e in path:
                loads[e] += w
                tot_time[e] += t
                tot_routes[e] += 1
                if return_cost_hist:
                    cost_hist[e].append(t)
    avg_time = {e:tt/tr for e,tt,tr in zip(tot_time,tot_time.values(),tot_routes.values()) if tr>0}
    nx.set_edge_attributes(G,loads,name)
    nx.set_edge_attributes(G,avg_time,name+'avgtime')
    if inplace: return (cost_hist if return_cost_hist else None)
    else: return (G,cost_hist if return_cost_hist else G)

def add_node_loads(G,path_mat,filter_areas=None,name='load', inplace=True):
    """
    Adds loads into nodes of a graph from an egde statistics list.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        Network's graphs.
    filter_areas: list like
        If provided, only consider paths originating in one of the zones in this list.
    name: string
        string to name the attribute in the graph
    inplace:  boolean
        If True, modifies the original graph.
    return_cost_hist: 
       If True, returns a histogram of the path's costs
    
    Returns
    -------
    NetworkX Graph and/or list
    """
    
    if not inplace:
        G = G.copy()
    loads = {}.fromkeys(G.nodes,0)
    tot_time = {}.fromkeys(G.nodes,0)
    tot_routes = {}.fromkeys(G.nodes,0)
    if return_cost_hist:
        cost_hist = {}.fromkeys(G.nodes)
        for n in cost_hist:
            cost_hist[n] = []
    for w,z,t,path,opp,pop,dist_decay in path_mat:
        if filter_areas is None or z in filter_areas:
            for e in path:
                loads[e[1]] += w
                tot_time[e[1]] += t
                tot_routes[e[1]] += 1
                if return_cost_hist:
                    cost_hist[e[1]].append(t)
    avg_time = {e:tt/tr for e,tt,tr in zip(tot_time,tot_time.values(),tot_routes.values()) if tr>0}
    nx.set_edge_attributes(G,loads,name)
    nx.set_edge_attributes(G,avg_time,name+'avgtime')
    
    if inplace: return (cost_hist if return_cost_hist else None)
    else: return (G,cost_hist if return_cost_hist else G)

def assess_scenarios(workers, *accessibility_scenarios):
    """
    Assess the accessibility of multiple scenarios (maps). Each scenario must 
    contain at least a GeoDataFrame of traffic zones (gdf) an associated graph (G) 
    and opportunities / population columns.
    
    Usage example:
    Calculate accessibility for multiple modes of transport (different graphs)
    assess_scenarios(workers,
                     {'G' : GraphWalk, 'gdf' : trafficZones,
                      'opportunities_column' : 'jobs',
                      'population_column' : 'pop'},
                      
                     {'G' : GraphDrive, 'gdf' : trafficZones,
                      'opportunities_column' : 'jobs',
                      'population_column' : 'pop'},
                      
                     {'G' : GraphBike, 'gdf' : trafficZones,
                      'opportunities_column' : 'jobs',
                      'population_column' : 'pop'},
                      
                     {'G' : GraphTransit, 'gdf' : trafficZones,
                      'opportunities_column' : 'jobs',
                      'population_column' : 'pop'})
    
    Parameters
    ----------
    workers : int 
        Number of threads. If -1 use all available.
    
    Returns
    -------
    list
    """
    return Parallel(n_jobs=workers)(delayed(accessibility)(**scenario)
                                    for scenario in accessibility_scenarios)


# these functions might be deprecated
def plot_element_hists(hists, elements, label='',plt_kws={'bins':20}):
    h=[]
    for el in elements:
        h+=hists[el]
    tot = ''.join(reversed(str(len(h))))
    
    tot = [''.join(reversed(tot[n*3:(n+1)*3])) for n in range(math.ceil(len(tot)/3))]
    tot = ' '.join(reversed(tot))
    plt.hist(h,label=f'{label} [{tot}]',**plt_kws)
    return None

def edge_statistics(G,path_mat,filter_areas=None,
                   name='load', inplace=True,
                   return_cost_hist=False):
    if not inplace:
        G = G.copy()
    loads = {}.fromkeys(G.edges,0)
    tot_time = {}.fromkeys(G.edges,0)
    tot_routes = {}.fromkeys(G.edges,0)
    if return_cost_hist:
        cost_hist = {}.fromkeys(G.edges)
        for e in cost_hist:
            cost_hist[e] = []
    for w,z,t,path,opp,pop,dist_decay in path_mat:
        if filter_areas is None or z in filter_areas:
            for e in path:
                loads[e] += w
                tot_time[e] += t
                tot_routes[e] += 1
                if return_cost_hist:
                    cost_hist[e].append(t)
    avg_time = {e:tt/tr for e,tt,tr in zip(tot_time,tot_time.values(),tot_routes.values()) if tr>0}
    nx.set_edge_attributes(G,loads,name)
    nx.set_edge_attributes(G,avg_time,name+'avgtime')
    if inplace: return (cost_hist if return_cost_hist else None)
    else: return (G,cost_hist if return_cost_hist else G)
