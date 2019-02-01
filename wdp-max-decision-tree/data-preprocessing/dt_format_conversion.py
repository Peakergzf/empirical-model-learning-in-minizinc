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

to the following 5 lists (each indexed by the nodes in bfs order)

1. feature index
[1, 3, -1, 2, -1, -1, -1]

2. relation
[LE, LE, LEAF, LE, LEAF, LEAF, LEAF]

3. feature value
[0.695117, 9.148, -1, 2.000000, -1, -1, -1]

    (the 3 lists above represent the left branching condition
     [self_cpi_min  <= 0.695117, all_cpi_mean  <= 9.148, true, self_cpi_mean  <= 2.000000, true, true, true] )

4. left child
[2, 4, -1, 6, -1, -1, -1]

5. leaf output value
[-1, -1, 1, -1, 1, 0, 1]

"""

NAMES = ["feature_idx", "feature_rel", "feature_val", "child", "val"]
LEAVES = [-1, "LEAF", -1, -1, -1]
DUMMIES = [-2, "DUM", -2, -2, -2]

LEAF_COND = "true"

FEAT_IDX = {
    "self_cpi_min": 1,
    "self_cpi_mean": 2,
    "all_cpi_mean": 3,
    "neigh_cpi_mean": 4,
}

FEAT_REL = {
    "<": "LT",
    "<=": "LE",
    ">": "GT",
    ">=": "GE",
    "==": "EQ",
    "=": "EQ",
}


class Output:
    def __init__(self, names):
        self.names = names
        self.cnt = len(self.names)
        self.output = [[] for _ in range(self.cnt)]

    def __str__(self):
        str_rep = ""
        for i in range(self.cnt):
            str_rep += self.names[i] + " = [|\n"
            for lst in self.output[i]:
                str_rep += str(lst).replace("'", '').replace('[', '').replace(']', '') + ",|\n"
            str_rep = str_rep.rstrip('\n').rstrip('|').rstrip(',') + " |];\n\n"
        return str_rep

    def add(self, inner_lists):
        for i in range(self.cnt):
            self.output[i].append(inner_lists[i])

    def fill_dummies(self):
        for i in range(self.cnt):
            max_len = max([len(lst) for lst in self.output[i]])
            for lst in self.output[i]:
                lst += [DUMMIES[i] for _ in range(max_len - len(lst))]


def read_file(file_name):
    """
    reads the decision trees from a file
    :return: a list of  trees
    """
    with open(file_name) as f:
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
    child = [LEAVES[NAMES.index("child")] for _ in range(n + 1)]
    val = [LEAVES[NAMES.index("val")] for _ in range(n + 1)]

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


def convert_cond(cond):
    """
    :param cond: condition (e.g. all_cpi_mean  <= 9.148 or true)
    :return: feat_idx, feat_rel, feat_val (e.g. 3, LE, 9.148 or -1, LEAF, -1)
    """
    feat_idx = [
        LEAVES[NAMES.index("feature_idx")]
        if c.split()[0] == LEAF_COND
        else FEAT_IDX[c.split()[0]]
        for c in cond
    ]
    feat_rel = [
        LEAVES[NAMES.index("feature_rel")]
        if c.split()[0] == LEAF_COND
        else FEAT_REL[c.split()[1]]
        for c in cond
    ]
    feat_val = [
        LEAVES[NAMES.index("feature_val")]
        if c.split()[0] == LEAF_COND
        else c.split()[2]
        for c in cond
    ]

    return feat_idx, feat_rel, feat_val


def convert_cond2(cond):
    """
    since all the left branching condition is LE, it can be emitted
    """
    feat_idx = [
        LEAVES[NAMES.index("feature_idx")]
        if c.split()[0] == LEAF_COND
        else FEAT_IDX[c.split()[0]]
        for c in cond
    ]

    feat_val = [
        LEAVES[NAMES.index("feature_val")]
        if c.split()[0] == LEAF_COND
        else c.split()[2]
        for c in cond
    ]

    return feat_idx, feat_val


def convert_tree_format(file_name):
    output = Output(NAMES)

    trees = read_file(file_name)

    for tree in trees:

        raw_edges = tree.split("\n")  # a list of  strings (each string represents an edge)
        n = len(raw_edges)  # n = the number of edges; n + 1 = the number of nodes

        if n == 1:  # (no branching/decision at all)
            output.add([[LEAVES[i]] if i != output.cnt - 1 else [int(raw_edges[0][-1])] for i in range(output.cnt)])
            continue

        edges = counting_sort_edges(raw_edges)

        node_num, has_child = pre_process_edges(edges)

        cond, child, val = construct_tree(n, edges, node_num, has_child)

        feat_idx, feat_rel, feat_val = convert_cond(cond)

        output.add([feat_idx, feat_rel, feat_val, child, val])

    output.fill_dummies()

    return str(output)
