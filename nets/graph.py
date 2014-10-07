def size(l):
    if hasattr(l, '__len__'):
        return len(l)
    else:
        return 1


def get_all_nodes(start_nodes):
    all_nodes = set(start_nodes)
    for node in start_nodes:
        if node.succ is not None:
            all_nodes = all_nodes.union(get_all_nodes(node.succ))
    return all_nodes


def traverse(node_list, prev_fn, next_fn, call_fn):
    deps = dict([(n, size(prev_fn(n))) for n in node_list])

    ready_nodes = [n for n in node_list if deps[n] == 0]
    while ready_nodes:
        next_nodes = list()
        for n in ready_nodes:
            call_fn(n)
            for s in next_fn(n):
                deps[s] -= 1
                if deps[s] == 0:
                    next_nodes.append(s)
            del deps[n]
        ready_nodes = next_nodes


def topological_traverse(node_list, back=False):
    # Forward prop

    prev_fn = lambda n: n.pred if hasattr(n.pred, '__len__') else [n.pred]
    next_fn = lambda n: n.succ if hasattr(n.succ, '__len__') else [n.succ]
    call_fn = lambda n: n.fprop()
    traverse(node_list, prev_fn, next_fn, call_fn)

    # Backprop

    if back:
        prev_fn, next_fn = next_fn, prev_fn
        call_fn = lambda n: n.bprop()
        traverse(node_list, prev_fn, next_fn, call_fn)
