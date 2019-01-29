
function var float: sigmoid(var float: a) ::promise_total =
    let { var 0.0..infinity: e2ma; % e^(-a)
          constraint float_exp(-a, e2ma);
          var 0.0..1.0: b = 1.0 / (1.0 + e2ma);
        } in b;

function var int: neural_network(
    int: n,
    int: hidden_layer_size,
    int: output_layer_size,
    array[int] of var float: x,
    array[int, int] of float: theta1,
    array[int, int] of float: theta2
) = 

let {

set of int: xRange = 1..n; 
set of int: aRange = 1..hidden_layer_size;
set of int: hRange = 1..output_layer_size;

set of int: xRangeExd = 1..n + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

array[xRangeExd] of var float: xExd = [1.0] ++ x;

array[aRange] of var float: a = [sigmoid(sum(j in xRangeExd)(xExd[j] * theta1[i, j])) | i in aRange];

array[aRangeExd] of var float: aExd = [1.0] ++ a;

array[hRange] of var float: h = [sigmoid(sum(j in aRangeExd)(aExd[j] * theta2[i, j])) | i in hRange];

% https://github.com/MiniZinc/libminizinc/commit/ee90a1f80ad717fa89d1a8cefe610ab4464966f5
% array[hRange] of var int: idx = [if h[i] - max(h) < 0.0001 then i else -1 endif | i in hRange];
% var int: pred_digit = if max(idx) == 10 then 0 else max(idx) endif;

% https://www.minizinc.org/doc-2.2.0/en/optiontypes.html
var int: pred_digit = [i | i in index_set(h) where h[i] == max(h)][1];

} in pred_digit;

















