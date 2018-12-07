m = 48  # the number of cores
LEAF_COND = "true"  # condition for leaf node
NO_CHILD = 0  # child for leaf node
NON_LEAF_VAL = -1  # classified value for non-leaf node


def read_file():
    with open("ptf_feat3.txt") as f:
        lines = f.read().rstrip()

    trees = lines.split("\n\n")  # a list of  strings (each string represents a tree)
    assert (len(trees) == m)  # one decision tree for each core

    return trees


def counting_sort_edges(raw_edges):
    depth = max([e.count("|") for e in raw_edges])

    edges = [[] for _ in range(depth + 1)]

    for e in raw_edges:
        indent = e.count("|")
        edges[indent].append(e)

    return edges


def pre_process_edges(edges):
    # outer list indexed by each level, inner list indexed by each node
    node_num = [[] for _ in range(len(edges))]  # bfs order
    has_child = [[] for _ in range(len(edges))]

    bfs_order = 2  # (skip the root node)

    for i in range(len(edges)):
        level = edges[i]
        for edge in level:
            node_num[i].append(bfs_order)
            bfs_order += 1
            has_child[i].append(":" not in edge)

    return node_num, has_child


def main():
    # built for MiniZinc model, each one is indexed by the number of cores
    conditions, children, values = [], [], []

    trees = read_file()

    for tree in trees:

        raw_edges = tree.split("\n")  # a list of  strings (each string represents an edge)
        n = len(raw_edges)  # n = the number of edges; n + 1 = the number of nodes

        if n == 1:  # (no branching/decision at all)
            conditions.append([LEAF_COND])
            children.append([NO_CHILD])
            values.append([raw_edges[0][-1]])  # e.g. get '1' from [ ': 1' ]
            continue

        edges = counting_sort_edges(raw_edges)

        node_num, has_child = pre_process_edges(edges)

        # indexed by each node in bfs order
        cond = []  # left branching condition for each node
        child = [NO_CHILD for _ in range(n + 1)]  # bfs order of the left child
        val = [NON_LEAF_VAL for _ in range(n + 1)]  # output value for leaf node

        # init root node
        child[0] = 2

        for i in range(len(edges)):
            level = edges[i]

            if i == 0:  # root node
                cond.append(level[i])
            else:
                edge_idx = 0
                for j in range(len(has_child[i - 1])):
                    if has_child[i - 1][j]:  # look up the previous level
                        cond.append(level[edge_idx].replace("|", "").strip()[:-3])
                        edge_idx += 2  # since binary
                    else:
                        cond.append(LEAF_COND)

            edge_idx = 0
            for j in range(len(level)):
                edge = level[j]
                idx = node_num[i][j] - 1  # convert from 1 to 0 based index
                if has_child[i][j]:
                    child[idx] = node_num[i + 1][edge_idx]  # next level
                    edge_idx += 2  # since binary
                else:
                    val[idx] = int(edge[-1])

        # add the deepest two leaf conditions
        cond += [LEAF_COND, LEAF_COND]

        conditions.append(cond)
        children.append(child)
        values.append(val)

        # testing
        if "2.114625" in tree:  # 2.114625  # 0.695117
            assert node_num == [[2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17], [18, 19], [20, 21]]
            assert has_child == [[True, True], [False, True, True, False], [True, False, True, False],
                                 [False, False, False, True], [True, False], [True, False], [False, False]]
            assert cond == ['self_cpi_min  <= 2.114625', 'self_cpi_min  <= 1.811556', 'self_cpi_min  <= 4.428', 'true',
                            'neigh_cpi_mean  <= 10.000', 'all_cpi_mean  <= 19.021', 'true',
                            'self_cpi_mean  <= 10.000000', 'true', 'all_cpi_mean  <= 2.680851', 'true', 'true', 'true',
                            'true', 'self_cpi_mean  <= 10.000', 'neigh_cpi_mean  <= 2.000', 'true',
                            'self_cpi_min  <= 3.394520', 'true', 'true', 'true']
            assert child == [2, 4, 6, 0, 8, 10, 0, 12, 0, 14, 0, 0, 0, 0, 16, 18, 0, 20, 0, 0, 0]
            assert val == [-1, -1, -1, 0, -1, -1, 1, -1, 1, -1, 1, 0, 1, 0, -1, -1, 1, -1, 1, 0, 1]


if __name__ == '__main__':
    main()
