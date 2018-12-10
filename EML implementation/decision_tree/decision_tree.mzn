
function var int: decision_tree(
  var int: i,
  var float: self_cpi_min,
  var float: self_cpi_mean,
  var float: all_cpi_mean,
  var float: neigh_cpi_mean,
  array[int] of var bool: cond,  % left branching condition for each node (true for leaf node)
  array[int] of int: child,      % bfs order of the left child for each node (-1 for leaf node)
  array[int] of int: val         % output value(class) for leaf node (-1 for non-leaf node)
) = 

if child[i] == -1 then val[i]
elseif cond[i] then decision_tree(child[i], self_cpi_min, self_cpi_mean, all_cpi_mean, neigh_cpi_mean, cond, child, val)
else decision_tree(child[i + 1], self_cpi_min, self_cpi_mean, all_cpi_mean, neigh_cpi_mean, cond, child, val)
endif;
