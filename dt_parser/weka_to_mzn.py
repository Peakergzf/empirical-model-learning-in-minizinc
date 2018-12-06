m = 48  # the number of cores
LEAF_COND = "true"  # condition for leaf node
NO_CHILD = 0  # child for leaf node
NON_LEAF_VAL = -1  # classified value for non-leaf node

with open("ptf_feat3.txt") as f:
    lines = f.read().rstrip()

trees = lines.split("\n\n")  # a list of  strings (each string represents a tree)
assert (len(trees) == m)  # one decision tree for each core

# build these lists for MiniZinc model, each one is indexed by the number of cores
conds, children, vals = [], [], []

for tree in trees:
    raw_edges = tree.split("\n")  # a list of  strings (each string represents an edge)
    n = len(raw_edges)  # n = the number of edges; n + 1 = the number of nodes

    if n == 1:  # (no branching/decision at all)
        conds.append([LEAF_COND])
        children.append([NO_CHILD])
        vals.append([raw_edges[0][-1]])  # e.g. get '1' from [ ': 1' ]
        continue

    # counting sort edges according to their bfs order

    depth = max([e.count("|") for e in raw_edges])

    edges = [[] for _ in range(depth + 1)]

    for e in raw_edges:
        indent = e.count("|")
        edges[indent].append(e)

    # build helper arrays

    node_num = [[] for _ in range(depth + 1)]
    has_child = [[] for _ in range(depth + 1)]

    bfs_order = 1  # used to number each node
    bfs_order += 1  # skip root node

    for i in range(len(edges)):
        level = edges[i]
        for edge in level:
            node_num[i].append(bfs_order)
            bfs_order += 1
            has_child[i].append(":" not in edge)

    # build ans arrays

    # cond = [None for _ in range(n + 1)]
    cond = []
    child = [None for _ in range(n + 1)]
    val = [None for _ in range(n + 1)]

    # TODO init for root node

    for i in range(len(edges)):
        level = edges[i]

        if i == 0:
            cond.append(level[i])
        else:
            next_j = 0
            for j in range(len(has_child[i - 1])):
                if has_child[i - 1][j]:
                    cond.append(edges[i][next_j].replace("|", "").strip()[:-3])
                    next_j += 2
                else:
                    cond.append(LEAF_COND)

        next_level_j = 0
        for j in range(len(level)):
            edge = level[j]
            idx = node_num[i][j] - 1  # convert from 1 to 0 based index
            if has_child[i][j]:
                child[idx] = node_num[i + 1][next_level_j]
                next_level_j += 2
            if not has_child[i][j]:
                val[idx] = edge[-1]

    # TODO
    cond += [LEAF_COND, LEAF_COND]

    conds.append(cond)
    children.append(child)
    vals.append(val)

    if "0.695117" in tree:  # 2.114625  # 0.695117
        # print(node_num)
        # print(has_child)
        print(cond)
        print(child)
        print(val)
