include "decision_tree.mzn";

% self_cpi_min  <= 0.695117
% |   all_cpi_mean  <= 9.148936
% |   |   self_cpi_mean  <= 2.000000: 0
% |   |   self_cpi_mean  > 2.000000: 1
% |   all_cpi_mean  > 9.148936: 1
% self_cpi_min  > 0.695117: 1

int: m = 7; % # of nodes
set of int: NODE = 1..m;
int: n = 4; % # of features
set of int: FEAT = 1..n;

enum REL = { LT, LE, GT, GE, EQ, LEAF, DUM };

array[FEAT] of var 0.0..20.0: feature;

array[NODE] of int: feature_idx = [1, 3, -1, 2, -1, -1, -1];

array[NODE] of REL: feature_rel = [LE, LE, LEAF, LE, LEAF, LEAF, LEAF];

array[NODE] of float: feature_val = [0.695117, 9.148936, -1, 2.000000, -1, -1, -1];

array[NODE] of int: child = [2, 4, -1, 6, -1, -1, -1];

array[int] of int: val = [-1, -1, 1, -1, 1, 0, 1];


constraint decision_tree( feature, feature_idx, feature_rel, feature_val, child, val ) = 1;

var float: obj = sum(i in FEAT)(feature[i]);

solve minimize obj;
