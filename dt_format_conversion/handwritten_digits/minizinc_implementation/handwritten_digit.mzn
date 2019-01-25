include "ann.mzn";

int: n;

int: hidden_layer_size;
int: output_layer_size;

set of int: xRange = 1..n; 
set of int: aRange = 1..hidden_layer_size;
set of int: hRange = 1..output_layer_size;

set of int: xRangeExd = 1..n + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

array[aRange, xRangeExd] of float: theta1;
array[hRange, aRangeExd] of float: theta2;

array[1..n] of float: x;


array[hRange] of var float: pred_digit == neural_network(
    n, 
    hidden_layer_size, 
    output_layer_size, 
    x, 
    theta1, 
    theta2
);

solve satisfy;

output [show(pred_digit)];

