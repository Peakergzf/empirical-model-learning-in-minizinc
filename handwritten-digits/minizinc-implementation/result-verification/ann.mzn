
function float: sigmoid(float: a) = 1.0 / (1.0 + exp(-a));


function int: neural_network(
    int: n,
    int: hidden_layer_size,
    int: output_layer_size,
    array[int] of float: x,
    array[int, int] of float: theta1,
    array[int, int] of float: theta2
) = 

let {

set of int: xRange = 1..n; 
set of int: aRange = 1..hidden_layer_size;
set of int: hRange = 1..output_layer_size;

set of int: xRangeExd = 1..n + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

array[xRangeExd] of float: xExd = [1.0] ++ x;

array[aRange] of float: a = [sigmoid(sum(j in xRangeExd)(xExd[j] * theta1[i, j])) | i in aRange];

array[aRangeExd] of float: aExd = [1.0] ++ a;

array[hRange] of float: h = [sigmoid(sum(j in aRangeExd)(aExd[j] * theta2[i, j])) | i in hRange];

array[hRange] of int: idx = [if h[i] - max(h) < 0.001 then i else -1 endif | i in hRange];

int: pred_digit = if max(idx) == 10 then 0 else max(idx) endif;

} in pred_digit;




















