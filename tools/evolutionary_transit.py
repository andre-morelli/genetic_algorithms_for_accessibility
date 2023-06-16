


def calc_boarding_cost(cycle, cars, ceil_to_closest=1,
                       min_hway=300):
    hway = cycle / cars
    # round up to closest multiple
    hway = ceil(hway / ceil_to_closest) * ceil_to_closest
    return min(hway, min_hway)

def update_log(G, log, route_cars, ceil_to_closest=1,
                      min_hway=300, copy = False):
    if copy:
        G = G.copy()
    for route, cars in route_cars.items():
        cycle = log[route]['cycle_time']
        b_edges = log[route]['boarding_edges']
        for e in b_edges:
            t = calc_boarding_cost(cycle, cars, 
                                  ceil_to_closest=ceil_to_closest,
                                  min_hway=min_hway)
            
            b_edges[e]['time_sec'] = t
        nx.set_edge_attributes(G, b_edges)
    return G