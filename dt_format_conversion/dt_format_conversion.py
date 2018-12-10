"""
Decision Tree Format Conversion

This module converts decision tree format

from

self_cpi_min  <= 0.695117
|   all_cpi_mean  <= 9.148936
|   |   self_cpi_mean  <= 2.000000: 0
|   |   self_cpi_mean  > 2.000000: 1
|   all_cpi_mean  > 9.148936: 1
self_cpi_min  > 0.695117: 1

to

[self_cpi_min  <= 0.695117, all_cpi_mean  <= 9.148, true, self_cpi_mean  <= 2.000000, true, true, true]
[2, 4, -1, 6, -1, -1, -1]
[-1, -1, 1, -1, 1, 0, 1]

representing: left branching condition, left child, output value for each node
(the nodes are numbered in their bfs order)
"""

# constants
LEAF_COND = "true"  # condition for leaf node
NO_CHILD = -1  # child for leaf node
NON_LEAF_VAL = -1  # classified value for non-leaf node


def read_file():
    """
    reads the decision trees from a file
    :return: a list of  trees
    """
    with open("ptf_feat3.txt") as f:
        lines = f.read().rstrip()

    trees = lines.split("\n\n")

    return trees


def counting_sort_edges(raw_edges):
    """
    counting sorts the edges in a tree according to their depth
    :param raw_edges: a list of edges
    :return: a list of list of edges (outer list: each level; inner list: each edge)
    """
    max_depth = max([e.count("|") for e in raw_edges])

    edges = [[] for _ in range(max_depth + 1)]

    for e in raw_edges:
        depth = e.count("|")
        edges[depth].append(e)

    return edges


def pre_process_edges(edges):
    """
    :param edges: a list of list of edges (outer list: each level; inner list: each edge)
    :return: node_num: bfs order for each node (except the root node)
                   has_child: whether has child or not for each node (except the root node)
    """
    node_num = [[] for _ in range(len(edges))]
    has_child = [[] for _ in range(len(edges))]

    bfs_order = 2  # (skip the root node)

    for i in range(len(edges)):
        level = edges[i]
        for edge in level:
            node_num[i].append(bfs_order)
            has_child[i].append(":" not in edge)
            bfs_order += 1

    return node_num, has_child


def construct_tree(n, edges, node_num, has_child):
    """
    :param n:  the number of edges
    :param edges: a list of list of edges (outer list: each level; inner list: each edge)
    :param node_num: bfs order for each node (except the root node)
    :param has_child: whether has child or not for each node (except the root node)
    :return: cond: left branching condition for each node
                   child: bfs order of the left child
                   val: output value for leaf node
    """
    cond = []
    child = [NO_CHILD for _ in range(n + 1)]
    val = [NON_LEAF_VAL for _ in range(n + 1)]

    # init root node
    fst_edge = edges[0][0]
    cond.append(fst_edge[:-3] if ':' in fst_edge else fst_edge)
    child[0] = 2

    for i in range(len(edges)):  # for each level
        left_edge_idx = 0  # index of edge as a left child within a level
        for j in range(len(edges[i])):  # for each edge
            idx = node_num[i][j] - 1  # convert from 1 to 0 based index
            if has_child[i][j]:
                edge = edges[i + 1][left_edge_idx].replace("|", "").strip()
                cond.append(edge[:-3] if ':' in edge else edge)
                child[idx] = node_num[i + 1][left_edge_idx]
                left_edge_idx += 2  # since binary
            else:
                val[idx] = int(edges[i][j][-1])
                cond.append(LEAF_COND)

    return cond, child, val


def format_output(results):
    """
    :param results: a list of strings
    :return: a combined formatted string
    """
    ans = ""
    for result in results:
        ans += str(result).replace("'", '').replace('[', '').replace(']', '') + ",|\n"
    ans = ans.rstrip('\n').rstrip('|').rstrip(',') + " |];\n\n"
    return ans


def create_data_file(conditions, children, values):
    """
    takes the result lists and write them into the output file
    """
    output = ""
    output += "cond = [|\n" + format_output(conditions)
    output += "child = [|\n" + format_output(children)
    output += "val = [|\n" + format_output(values)

    # clear the output file
    with open("data.dzn", 'w'):
        pass

    with open("data.dzn", 'a') as data_file:
        data_file.write(output)


def main():
    # each indexed by the number of cores
    conditions, children, values = [], [], []

    trees = read_file()

    for tree in trees:

        raw_edges = tree.split("\n")  # a list of  strings (each string represents an edge)
        n = len(raw_edges)  # n = the number of edges; n + 1 = the number of nodes

        if n == 1:  # (no branching/decision at all)
            conditions.append([LEAF_COND])
            children.append([NO_CHILD])
            values.append([int(raw_edges[0][-1])])  # e.g. get '1' from [ ': 1' ]
            continue

        edges = counting_sort_edges(raw_edges)

        node_num, has_child = pre_process_edges(edges)

        cond, child, val = construct_tree(n, edges, node_num, has_child)

        conditions.append(cond)
        children.append(child)
        values.append(val)

        # testing
        if "2.114625" in tree:  # 2.114625  # 0.695117
            assert node_num == [[2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17], [18, 19], [20, 21]]
            assert has_child == [[True, True], [False, True, True, False], [True, False, True, False],
                                 [False, False, False, True], [True, False], [True, False], [False, False]]
            assert cond == ['self_cpi_min  <= 2.114625', 'self_cpi_min  <= 1.811556',
                            'self_cpi_min  <= 4.428939', 'true', 'neigh_cpi_mean  <= 10.000000',
                            'all_cpi_mean  <= 19.021277', 'true', 'self_cpi_mean  <= 10.000000',
                            'true', 'all_cpi_mean  <= 2.680851', 'true', 'true', 'true', 'true',
                            'self_cpi_mean  <= 10.000000', 'neigh_cpi_mean  <= 2.000000',
                            'true', 'self_cpi_min  <= 3.394520', 'true', 'true', 'true']
            assert child == [2, 4, 6, -1, 8, 10, -1, 12, -1, 14, -1, -1, -1, -1, 16, 18, -1, 20, -1, -1, -1]
            assert val == [-1, -1, -1, 0, -1, -1, 1, -1, 1, -1, 1, 0, 1, 0, -1, -1, 1, -1, 1, 0, 1]

    max_cond = max([len(cond) for cond in conditions])
    for cond in conditions:
        cond += ['false' for _ in range(max_cond - len(cond))]

    max_child = max([len(child) for child in children])
    for child in children:
        child += ['-2' for _ in range(max_child - len(child))]

    max_val = max([len(val) for val in values])
    for val in values:
        val += ['-2' for _ in range(max_val - len(val))]

    create_data_file(conditions, children, values)


if __name__ == '__main__':
    main()
