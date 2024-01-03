def check_valid_graph(nodes, edge_idx):
    # nodes: Dictionary of node_id to Node object
    # edge_idx: List of 2 lists, from_idx and to_idx, each of which is a list of node_ids
    assert(type(nodes) == dict)
    assert(type(edge_idx) == list)
    nodeids = [int(node_id) for node_id in nodes]
    nodeids = set(nodeids)
    edgeids = set()
    for from_idx, to_idx in zip(edge_idx[0], edge_idx[1]):
        edgeids.add(int(from_idx))
        edgeids.add(int(to_idx))

    return edgeids.issubset(nodeids)