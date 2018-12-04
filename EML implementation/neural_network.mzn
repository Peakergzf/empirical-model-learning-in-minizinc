
function var float: neural_network(
  int: input_layer_size, 
  int: hidden_layer_size, 
  array[int] of var float: x, % input layer
  array[int, int] of float: theta1, % from input layer to hidden layer
  array[int] of float: theta2 % from hidden layer to hypothesis
) = 

let {

set of int: xRange = 1..input_layer_size; 
set of int: aRange = 1..hidden_layer_size;
% with bias unit
set of int: xRangeExd = 1..input_layer_size + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

% hidden layer
array [aRangeExd] of var float: a;

constraint forall(i in aRangeExd)(
    if i == hidden_layer_size + 1 then a[i] = 1 % bias unit
    else a[i] = tanh(sum(j in xRangeExd)(x[j] * theta1[i - 1, j]))
    endif
);

% hypothesis
var float: h;

constraint h = tanh(sum(i in aRangeExd)(a[i] * theta2[i]));

} in h;
