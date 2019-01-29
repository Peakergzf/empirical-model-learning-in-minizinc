include "globals.mzn";

function var float: sigmoid(var float: a) ::promise_total =
let { 
var 0.0..infinity: e2ma; % e^(-a)
constraint float_exp(-a, e2ma);
var 0.0..1.0: b = 1.0 / (1.0 + e2ma);
} in b;

function var int: neural_network(
    int: n, % input layer size
    int: hidden_layer_size,
    int: output_layer_size,
    array[int] of var float: x, % input features
    array[int, int] of float: theta1,
    array[int, int] of float: theta2
) = 

let {

set of int: xRange = 1..n; 
set of int: aRange = 1..hidden_layer_size;
set of int: hRange = 1..output_layer_size;

set of int: xRangeExd = 1..n + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

% add bias unit
array[xRangeExd] of var float: xExd = [1.0] ++ x;

% from input layer to hidden layer
array[aRange] of var float: a = [sigmoid(sum(j in xRangeExd)(xExd[j] * theta1[i, j])) | i in aRange];

% add bias unit
array[aRangeExd] of var float: aExd = [1.0] ++ a;

% from hidden layer to output layer
array[hRange] of var float: h = [sigmoid(sum(j in aRangeExd)(aExd[j] * theta2[i, j])) | i in hRange];

% the digit with the highest probability
} in arg_max(h);

