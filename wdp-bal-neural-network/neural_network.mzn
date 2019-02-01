
function var float: mytanh(var float: a) ::promise_total =
    let { var 0.0..infinity: e2a;
          constraint float_exp(2.0*a,e2a);
          var 0.0..1.0: b = (e2a-1.0)/(e2a+1.0);
        } in b;

function var float: neural_network(
  int: input_layer_size, 
  int: hidden_layer_size, 
  array[int] of var float: x,       % input layer
  array[int, int] of float: theta1, % from input layer to hidden layer
  array[int] of float: theta2       % from hidden layer to hypothesis
) = 

let {

constraint assert(length(x) == input_layer_size, "inconsistent input layer size");
constraint assert(length(theta2) == hidden_layer_size + 1, "inconsistent theta2 size");

set of int: xRange = 1..input_layer_size; 
set of int: aRange = 1..hidden_layer_size;
% with bias unit
set of int: xRangeExd = 1..input_layer_size + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

array[xRangeExd] of var float: xExd = x ++ [1.0];

% hidden layer    
array [aRange] of var 0.0..1.0: a = [ mytanh(sum(j in xRangeExd)(xExd[j] * theta1[i, j])) | i in aRange ];

array[aRangeExd] of var float: aExd = a ++ [1.0];

} in mytanh(sum(i in aRangeExd)(aExd[i] * theta2[i])); % hypothesiss
