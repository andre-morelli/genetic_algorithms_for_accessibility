import osmnx as ox
import os
import zipfile
import pandas as pd
from tempfile import TemporaryDirectory
import networkx as nx
import numpy as np
from math import radians, degrees, sin, cos, asin, acos, sqrt, ceil
import warnings

def read_gtfs_as_dict(path, get_only=None):
    """
    Reads a GTFS as a dictionay with all tables.

    Parameters
    ----------
    path: string
        path of the feed
    get_only: list like
        A list of GTFS standard file names such as 
        ['frequencies', 'stops', <...>]
        If provided, only the files in the list will be read.
         
    
    Returns
    -------
    dictionary
    """
    
    data = {}
    with TemporaryDirectory() as temp_dir:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                path = temp_dir

        for file in os.listdir(path):
            if file.endswith('.txt'):
                fname = file.replace('.txt','')
                if get_only is not None and fname not in get_only:
                    continue
                file_path = os.path.join(path, file)
                data[file.replace('.txt','')] = pd.read_csv(file_path, encoding="utf-8-sig")

    return data

def _convert_time_string(df,cols=['departure_time','arrival_time']):
    """
    Helper function for :func:`read_merged_gtfs_files`.
    It converts departure and arrival times into flaots 
    """
    
    df=df.copy()
    for col in cols:
        df[col] = [(int(n.split(':')[0])*3600+
                    int(n.split(':')[1])*60+
                    int(n.split(':')[2]))
                   for n in df[col]]
    return df

def _get_frequency_df(feed,start,stop):
    """
    Helper function for :func:`read_merged_gtfs_files`.
    It reads the frequency table for the GTFS (which is
    not always present) 
    """
    freqs = _convert_time_string(feed['frequencies'], cols = ['start_time','end_time'])
    freqs = freqs[(freqs['start_time']>=start)&
                  (freqs['end_time']<=stop)]
    
    freqs = freqs.merge(feed['trips'], on='trip_id').merge(feed['routes'], on='route_id')
    return freqs

def _no_oulier_mean(x):
    """
    Helper function for :func:`get_transit_lines_as_graphs`.
    A mean ignoring outliers. It is useful for defining the
    average frequency of a route without the interference of
    rare events.
    """
    
    if len(x) == 0:
        return None
    #turn into array
    x=np.array(x)
    #calculate inter quartile range (IQR)
    q1, q3 = np.percentile(x,25), np.percentile(x,75)
    inter_quartile_dist = q3 - q1
    # use IQR to filter outliers out
    x=x[(x>=q1-1.5*inter_quartile_dist)&
        (x<=q3+1.5*inter_quartile_dist)]
    filt_mean=(np.mean(x) if len(x)>0 else mean)
    return filt_mean

def _calc_headway(route_id, freqs):
    """
    Helper function for :func:`get_transit_lines_as_graphs`.
    Calculate the average headway of the route.
    """
    
    if len(freqs[freqs['route_id']==route_id])==0:
        return None
    else:
        return _no_oulier_mean(freqs[freqs['route_id']==route_id]['headway_secs'])

def _clean_shortcuts(L):
    """
    Helper function for :func:`get_transit_lines_as_graphs`.
    Cleans eventual shortcuts in the transit line.
    Some feeds have buses that skip some stops at time, configuring a
    shortcut in the system. This means that, unless the user is going
    to the stop being cut, the system will appear to always take the 
    shortest route in the final graph, which is not the case in 
    reality.
    """
    
    while True:
        for node in L.nodes:
            ns = list(nx.neighbors(L,node))
            if len(ns)>1:
                tmax=0
                for n in ns:
                    t=L.edges[(node,n,0)]['time_sec']
                    if t>tmax:
                        tmax=t
                        remove=(node,n,0)
                L.remove_edge(*remove)
                break
        else:
            break
    return L

def read_merged_gtfs_files(path, start=6*3600, stop=9*3600,
                           day_of_the_week='wednesday'):
    """
    Reads a GTFS as a single DataFrame with merged information.

    Parameters
    ----------
    path: string
        path of the feed
    start: float
        Time of day in seconds. Will only consider routes opperanting
        from this time
    stop: float
        Time of day in seconds. Will only consider routes opperanting
        before this time
    day_of_the_week: string
        will only consider routes opporating at this day
         
    
    Returns
    -------
    merged DataFrame, frequencies DataFrame and GTFS dictionary
    """
    
    gtfs = read_gtfs_as_dict(path)
    complete = gtfs['stop_times']
    for df,col in [(gtfs['trips'], 'trip_id'),
                   (gtfs['routes'], 'route_id'),
                   (gtfs['stops'], 'stop_id')]:
        complete = complete.merge(df, on=col)
    complete = _convert_time_string(complete)
    
    # the wednesday rule!!!!
    wed = gtfs['calendar'][( gtfs['calendar'][day_of_the_week]==1) ]['service_id'].unique()
    complete = complete[ (complete['service_id'].isin(wed)) ]
    
    #frequencies if possible
    if 'frequencies' in gtfs:
        freq = _get_frequency_df(gtfs, start, stop)
        routes = freq['trip_id'].unique()
        complete = complete[ complete['trip_id'].isin(routes) ]
    else:
        freq=None
    return complete, freq, gtfs



def read_gtfs_as_graphs(gtfs_path, start=6*3600, stop=9*3600,
                         day_of_the_week = 'wednesday',
                         clean_shrtcts=True, 
                         interpolate_stop_times=False,
                         rest_per_cycle=600,
                         infer_headway=False):
    """
    Reads a GTFS as a single DataFrame with merged information.

    Parameters
    ----------
    gtfs_path: string
        path of the feed
    start: float
        Time of day in seconds. Will only consider routes opperanting
        from this time
    stop: float
        Time of day in seconds. Will only consider routes opperanting
        before this time
    clean_shrtcts: boolean
        If True, removes shortcuts from routes, preventing them skipping
        stops
    interpolate_times: boolean
        NOT IMPLEMENTED. If True, interpolates time between stops
    rest_per_cycle: int
        Seconds added to the each end of the cycle time to account for resting 
        and schedule corrections (mostly applicable to buses)
         
    
    Returns
    -------
    list of NetworkX DiGraphs
    """
    
    complete,freq,feed = read_merged_gtfs_files(gtfs_path)
    routes = []
    for rid in complete['route_id'].unique():
        temp = complete[ complete['route_id']==rid ]
        # route identification
        name = temp.iloc[0]['route_short_name']
        lname = temp.iloc[0]['route_long_name']
        color = temp.iloc[0]['route_color']
        mode = temp.iloc[0]['route_type']
        
        #get nodes
        
        temp = temp.sort_values(['trip_id','stop_sequence'])
        ns = temp['stop_id'].unique()
        nodes = {}
        for sid in ns:
            dat = feed['stops'].loc[feed['stops']['stop_id']==sid].iloc[0]
            nodes[f'{rid}_{sid}'] = {'x' : dat['stop_lon'],
                                     'y' : dat['stop_lat'],
                                     'name' : dat['stop_name']}
        if len(nodes)<2:
            continue
        L = nx.MultiDiGraph()
        L.add_nodes_from(nodes)
        nx.set_node_attributes(L,nodes)
        
        # create dictionaries for cycles and edges
        edges={}
        cycles={}
        for tid in temp['trip_id'].unique():
            temp2 = temp[temp['trip_id']==tid]
            if len(temp2)<2:
                continue
            #get cycle time as the difference between start and finish
            if temp2.iloc[0]['stop_sequence']==1:
                cycles[temp2.iloc[0]['direction_id']] = (cycles.get(temp2.iloc[0]['direction_id'],[])+
                                                         [(temp2.iloc[-1]['departure_time'] -
                                                         temp2.iloc[0]['departure_time']) + 
                                                         rest_per_cycle])
                
            times = (np.array(temp2['departure_time'])[1:]-
                     np.array(temp2['departure_time'])[:-1])
            # get distances if possible
            if 'shape_dist_traveled' in temp2.columns:
                dists = (np.array(temp2['shape_dist_traveled'])[1:]-
                         np.array(temp2['shape_dist_traveled'])[:-1])
            else:
                dists = [np.nan]*len(times)
            es = zip(temp2['stop_id'].to_list()[:-1],
                     temp2['stop_id'].to_list()[1:],
                     times,dists)
            
            for e0,e1,t,d in es:
                e0,e1 = f'{rid}_{e0}',f'{rid}_{e1}'
                if (e0,e1,0) in edges:
                    edges[(e0,e1,0)][0].append(t)
                    edges[(e0,e1,0)][1].append(d)
                else:
                    edges[(e0,e1,0)]=([t],[d])
        # update info into edge dictionary
        for e in edges:
            t,d=edges[e]
            edges[e]={'time_sec' : np.mean(t),
                      'route_id' : rid,
                      'length' : np.mean(d),
                      'short_name' : name,
                      'long_name' : lname,
                      'color' : color,
                      'mode' : mode}
        # add info to graph
        L.add_edges_from(edges)
        nx.set_edge_attributes(L,edges)
        
        #clean up?
        if clean_shrtcts:
            _clean_shortcuts(L)
        
        #estimate cycle
        c = 0
        for v in cycles.values():
            c+=np.mean(v)
        
        # get headway if frequencies exist
        if freq is not None:
            hway = _calc_headway(rid,freq)
        
        # else try to infer frequency
        
        elif infer_headway:
            times=[]
            for node in L.nodes:
                f = temp[ temp['stop_id'] == node.replace(f'{rid}_','') ]
                f = f.sort_values('departure_time')
                if len(f)>1:
                    times.append(np.mean(np.array(f['departure_time'])[1:]-
                                         np.array(f['departure_time'])[:-1]))
            hway=_no_oulier_mean(times)
         
        else:
            hway=None   
        # add graph characteristiscs and make it compatible with OSMnx
        L.graph={'route_id': rid,
                 'mode' : mode,
                 'name' : name,
                 'crs' : 'epsg:4326',
                 'cycle_time' : c,
                 'headway' : hway}
        routes.append(L)
        
    return routes


#################################################################################
# In this section we present some functions to create multigraphs with OSMnx

def _great_circle(lon1, lat1, lon2, lat2):
    """
    Helper function for :func:`get_closest_nodes`.
    Calculates great circle distance from coordinates to apply tolerance.
    """
    
    if lon1==lon2 and lat1==lat2:
        return 0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6.371e6 * (
        acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2))
    )

def get_closest_nodes(G, routes, tol=100):
    """
    Updates graphs in "routes" to create an "attached" attribute.
    Nodes of the route will attach to the original graph (G) in these
    nodes in :func:`add_transit_route`.

    Parameters
    ----------
    G: NetworkX Graph from OSMnx
        Graph of the region where you wish to add transit routes
    routes: list
        A list of transit routes in graph format. Generally, is the 
        result from :func:`get_transit_routes_as_graphs` 
    tol: float
        Tolerance in meters. If there is no pair of nodes closer than 
        this, the graphs will not be attached at this point
         
    
    Returns
    -------
    None
    """
    
    #can two points have the same node? 
    node_names={}
    X, Y = [], []
    routes_coords = [0] + [len(sub) for sub in routes]
    routes_coords = np.cumsum(routes_coords)
    
    for sub in routes:
        for node in sub.nodes:
            x0, y0 = sub.nodes[node]['x'], sub.nodes[node]['y']
            X.append(x0)
            Y.append(y0)
    
    ns = ox.get_nearest_nodes(G, np.array(X), np.array(Y),
                              method='balltree')
    
    for sub, pos0, pos1 in zip(routes, routes_coords[:-1], routes_coords[1:]):
        node_names = {}
        for node,name in zip(sub.nodes, ns[pos0:pos1]):
            d = _great_circle(G.nodes[name]['x'], G.nodes[name]['y'],
                              sub.nodes[node]['x'], sub.nodes[node]['y'])
            if d<=tol:
                node_names[node] = name
            else:
                node_names[node] = None
        nx.set_node_attributes(sub, node_names, 'attached')
    return None

def add_transit_route(G, L, route_name=None, headway=None, hway_min=5,
                      buses=None, avg_stop_time=0, copy=False,
                      attach_node_attr='attached', log=None):
    """
    Adds the transit route's graph to the base graph (G).
    The costs involved in boarding/unboarding can be set
    with specific rules (buses or headway)

    Parameters
    ----------
    G: NetworkX Graph from OSMnx
        Graph of the region where you wish to add transit routes
    L: NetworkX Graph
        A transit graph. Generally, is the result from 
        :func:`get_transit_routes_as_graphs`
    route_name:
        Name or identification of the route
    headway: float
        Time between two buses for this route
    hway_min: float
        The minimum headway possible
    buses: int
        Number of buses or transit cars opperating in this route
    avg_stop_time: float
        Average time stoped at each stop. If cycle time is precise,
        set this to zero.
    attach_node_attr: str
        name of the attribute in L that designates the attached node in
        G. Result from :func: `get_closest_nodes`
    copy: boolean
        If True, returns a copy of the original graph, else, change the original 
        graph (if True, it adds some computational time, particularly for large
        graphs)
    log: dictionary or None
        If provided, updates the dictionary to include the edges added to the
        graph. This function is useful for optimization processes that require
        constant updating of the graph
    
    Returns
    -------
    NetworkX graph
    """
    
    if route_name is None:
        route_name = L.graph['name']
    
    if attach_node_attr not in list(L.nodes(data=True))[0][1]:
        warnings.warn('The original graph has no attribute to attach with. ' +
                      'An attachment attribute will be created. This might ' + 
                      'take time, so consider using "get_closest_nodes" ' +
                      'before using this function')
        rs = [L]
        get_closest_nodes(G, rs) # tolearnce is 100 m
        L = rs[0]
    
    L = L.copy()
    mode = list(L.edges(data='mode'))[0][2]
    if buses is None and headway is None:
        if L.graph['headway'] is not None:
            headway = L.graph['headway']
        else:
            raise ValueError('No headway found. Buses or headway must be set')
    
    if headway is None:
        if buses>0:
            T = (sum(l for _,_,l in L.edges(data='time_sec')) +
                     len(L)*avg_stop_time)/buses
            if T<hway_min:
                T = hway_min
        else:
            T = 1e30
        headway=T
    
    if copy:
        G = G.copy()
        
    G.update(L)
    edges = {}
    for n in L:
        pair = L.nodes[n][attach_node_attr]
        if pair not in (G.nodes):
            continue
        #entering transit system:
        edges[(pair,n,0)]=dict(time_sec=headway/2, length=.01, mode=mode,
                               route_id=route_name, etype='boarding')
        #leaving transit system:
        edges[(n,pair,0)]=dict(time_sec=.1, length=.01, mode=mode, 
                               route_id=route_name, etype='unboarding')
    if log is not None:
        log[route_name] = {
                'cycle_time' : L.graph['cycle_time'],
                'boarding_edges' : {e:dat for e, dat in edges.items() if dat['etype']=='boarding'},
                'unboarding_edges' : {e:dat for e, dat in edges.items() if dat['etype']=='unboarding'},
                'route_edges' : {(e0, e1, k) : dat for e0, e1, k, dat in L.edges(data=True, keys=True)}
            }
    G.add_edges_from(edges)
    nx.set_edge_attributes(G,edges)
    
    if not 'transit_route' in G.graph:
        G.graph['transit_routes']={}
    G.graph['transit_routes'][route_name]={'headway':headway,
                                          'avg_stop_time':avg_stop_time}
    
    return G

def get_cars_per_route(routes, mode=None):
    """
    A function to estimate how many cars each route has in opperation

    Parameters
    ----------
    routes: list of Graphs
        The graphs of routes in the system
    mode: 
        id of the mode of transport
    
    Returns
    -------
    dictionary
    """
    
    cars = {}
    for L in routes:
        if mode is not None:
            if L.graph['mode'] != mode:
                continue
        cars[L.graph['route_id']] = ceil(L.graph['cycle_time'] / L.graph['headway'])
    return cars


# this is particular to genetica algorithms
def assign_buses(routes, bus_gene, one_at_least=False, mode=None):
    if one_at_least:
        if mode is None:
            buses={L.graph['route_id']:1 for L in routes}
        else:
            buses={L.graph['route_id']:1 for L in routes if L.graph['mode']==mode}
    else:
        if mode is None:
            buses={L.graph['route_id']:0 for L in routes}
        else:
            buses={L.graph['route_id']:0 for L in routes if L.graph['mode']==mode}
    for g in bus_gene:
        buses[g]+=1
    for L in routes:
        if mode is not None:
            if L.graph['mode'] != mode:
                continue
        if buses[L.graph['route_id']]!=0:
            L.graph['headway']=ceil(L.graph['cycle_time']/buses[L.graph['route_id']]/300)*300
        else:
            L.graph['headway']=1e30
    return buses
