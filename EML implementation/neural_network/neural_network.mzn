
function var float: neural_network(
  int: input_layer_size, 
  int: hidden_layer_size, 
  array[int] of var float: x,       % input layer
  array[int, int] of float: theta1, % from input layer to hidden layer
  array[int] of float: theta2       % from hidden layer to hypothesis
) = 

let {

constraint assert(length(x) == input_layer_size + 1, "inconsistent input layer size");
constraint assert(length(x) == input_layer_size + 1, "inconsistent theta1 size");
constraint assert(length(theta2) == hidden_layer_size + 1, "inconsistent theta2 size");

set of int: xRange = 1..input_layer_size; 
set of int: aRange = 1..hidden_layer_size;
% with bias unit
set of int: xRangeExd = 1..input_layer_size + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

% hidden layer
array [aRangeExd] of var float: a = [
    if i == hidden_layer_size + 1 then 1 % bias unit
    else tanh(sum(j in xRangeExd)(x[j] * theta1[i, j]))
    endif
    | i in aRangeExd ];

} in tanh(sum(i in aRangeExd)(a[i] * theta2[i])); % hypothesiss
