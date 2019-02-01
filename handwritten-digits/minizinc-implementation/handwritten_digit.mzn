include "neural_network.mzn";

% ================================== IMAGE ==========================================

% (20 x 20 pixel grayscale image, flatten into 1d array of size 400)

int: n; %  = 400

array[1..n] of float: original;

array[1..n] of var -10..10: noise;

float: factor = 0.01;

array[1..n] of var float: image = [ original[i] + noise[i] * factor |  i in 1..n ];

% ================================== NEURAL NETWORK =================================

% input_layer_size = n = 400
int: hidden_layer_size;  % = 25
int: output_layer_size;  % = 10

set of int: xRange = 1..n; 
set of int: aRange = 1..hidden_layer_size;
set of int: hRange = 1..output_layer_size;

set of int: xRangeExd = 1..n + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

% weights for the neural network
array[aRange, xRangeExd] of float: theta1;
array[hRange, aRangeExd] of float: theta2;

% -----------------------------------------------------------------------------------

% the original image is digit 0 (output by the ann as 10)

% minimize the noise s.t. the ann can't correctly recognize the image anymore 
% or
% maximize the noise s.t. the ann can still correctly recognize the image

constraint 10 != neural_network(
    n, 
    hidden_layer_size, 
    output_layer_size, 
    image, 
    theta1, 
    theta2
);

var int: obj = sum(noise);

solve minimize obj;

