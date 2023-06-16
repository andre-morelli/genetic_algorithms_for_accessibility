import osmnx as ox
import igraph as ig
import networkx as nx
import math

def get_igraph(G, edge_weights=None, node_weights=None):
    """
    Transforms a NetworkX graph into an iGraph graph.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be converted.
    edge_weights: list or string
        weights stored in edges in the original graph to be kept in new graph. 
        If None, no weight will be carried. See get_full_igraph to get all 
        weights and attributes into the graph.
    node_weights: list or string
        weights stored in nodes in the original graph to be kept in new graph. 
        If None, no weight will be carried. See get_full_igraph to get all 
        weights and attributes into the graph.

    Returns
    -------
    iGraph graph
    """
    if type(edge_weights) == str:
        edge_weights = [edge_weights]
    if type(node_weights) == str:
        node_weights = [node_weights]
    n_names = list(G.nodes)
    e_names = list(G.edges)
    G = nx.relabel.convert_node_labels_to_integers(G)
    Gig = ig.Graph(directed=True)
    Gig.add_vertices(list(G.nodes()))
    Gig.add_edges(list(G.edges()))
    if 'kind' not in G.graph.keys():
        G.graph['kind']='primal' # if not specified, assume graph id primal
    if G.graph['kind']=='primal':
        Gig.vs['name'] = n_names
    elif G.graph['kind']=='dual':
        Gig.vs['name'] = e_names
    if edge_weights != None:
        for weight in edge_weights:
            Gig.es[weight] = [n for _,_,n in G.edges(data=weight)]
            
    if node_weights != None:
        for weight in node_weights:
            Gig.vs[weight] = [n for _,n in G.nodes(data=weight)]
    return Gig
    

def get_full_igraph(G):
    """
    Transforms a NetworkX graph into an iGraph graph keeping all possible info.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be converted.
    
    Returns
    -------
    iGraph graph
    """
    all_edge_attrs = []
    all_node_attrs = []
    for edge in G.edges:
        for attr in G.edges[edge].keys():
            if attr in all_edge_attrs:
                continue
            all_edge_attrs.append(attr)
                
    for node in G.nodes:
        for attr in G.nodes[node].keys():
            if attr in all_node_attrs:
                continue
            all_node_attrs.append(attr)       
    
    return get_igraph(G, all_edge_attrs, all_node_attrs)
    
def get_dual(G, node_to_edge='first'):
    """
    Transforms a NetworkX primal graph into a NetworkX dual graph.
    This function keeps edge properties in the nodes of dual graph
    and adds angular weights, keeping the x and y central positions 
    of edges so this graph can be plotted with OSMnx. 
    
    obs: This function is partially based on previous work by 
    Alceu Dal Bosco Jr.

    Parameters
    ----------
    G : NetworkX DiGraph or Graph 
        The graph to be converted.
    node_to_edge: 'first','last' or None
        gets attributes from nodes of the original graph to add to dual
        graph. 'first' gets the attribute from the first node of the edge, 
        'last' gets from the last. If None, ignore orginal graphs node
        attributes
    Returns
    -------
    NetworkX graph
    """
    
    Gd = nx.line_graph(G)
    Gd.graph = G.graph
    all_edge_attrs = []
    all_node_attrs = []
    for edge in G.edges:
        for attr in G.edges[edge].keys():
            if attr in all_edge_attrs:
                continue
            all_edge_attrs.append(attr)
    for node in G.nodes:
        for attr in G.nodes[node].keys():
            if attr in all_node_attrs:
                continue
            all_node_attrs.append(attr)

    for node in Gd.nodes:
        if node_to_edge=='first':
            for attr in all_node_attrs:
                if attr in ['x', 'y']:
                    continue
                try:
                    Gd.nodes[node][attr] = G.nodes[node[0]][attr]
                except:
                    Gd.nodes[node][attr] = None
        for attr in all_edge_attrs:
            try:
                Gd.nodes[node][attr] = G.edges[node][attr]
            except:
                    Gd.nodes[node][attr] = None

    G_ = ox.project_graph(G)
    for e in Gd.nodes:
        Gd.nodes[e]['x'] = (G.nodes[e[0]]['x']+G.nodes[e[1]]['x'])/2
        Gd.nodes[e]['y'] = (G.nodes[e[0]]['y']+G.nodes[e[1]]['y'])/2
        #Gd.nodes[e]['osmid'] = (G.nodes[e[0]], G.nodes[e[1]])
    for e1,e2,l in Gd.edges:
        ang_dif = _dif_angle((G_.nodes[e1[0]]['x'],G_.nodes[e1[0]]['y']), 
                            (G_.nodes[e1[1]]['x'],G_.nodes[e1[1]]['y']), 
                            (G_.nodes[e2[1]]['x'],G_.nodes[e2[1]]['y']))
        if ang_dif == 0:
            ang_dif += .0001 #avoid null values
        Gd.edges[(e1,e2,l)]['ang_dif'] = ang_dif
    Gd.graph['kind'] = 'dual'
    
    #quick and dirty fix for new OSMnx indexing of osmids
    mapping = {n:str(n) for n in Gd.nodes}
    Gd = nx.relabel_nodes(Gd,mapping)
    return Gd
    
def _dif_angle(a, b, c):
    """
    Gets angle difference

    Parameters
    ----------
    a, b, c : tuples
        sequential node coords on projected graph.
    
    Returns
    -------
    float
    """
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return abs(ang) - 180 if abs(ang) > 180 else 180 - abs(ang)