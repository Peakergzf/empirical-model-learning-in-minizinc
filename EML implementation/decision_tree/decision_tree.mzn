
function var int: dt_aux(
  int: i,
  array[int] of var bool: cond,
  array[int] of int: child,
  array[int] of int: val
) = 
if child[i] == -1 then val[i]
elseif cond[i] then dt_aux(child[i], cond, child, val)
else dt_aux(child[i + 1], cond, child, val)
endif;



function var int: decision_tree(
  array[int] of var float: feature,
  array[int] of int: feature_idx,
  array[int] of REL: feature_rel,
  array[int] of float: feature_val,
  array[int] of int: child,
  array[int] of int: val
) = 
let {
  array[int] of var bool: cond = [
    if feature_rel[i] = LEAF then true
    elseif feature_rel[i] = DUM then false
    elseif feature_rel[i] = LT then feature[feature_idx[i]] <  feature_val[i]
    elseif feature_rel[i] = LE then feature[feature_idx[i]] <= feature_val[i]
    elseif feature_rel[i] = GT then feature[feature_idx[i]] >  feature_val[i]
    elseif feature_rel[i] = GE then feature[feature_idx[i]] >= feature_val[i]
    else feature[feature_idx[i]] == feature_val[i]
    endif
    | i in index_set(feature_idx)];
} in dt_aux(1, cond, child, val)
