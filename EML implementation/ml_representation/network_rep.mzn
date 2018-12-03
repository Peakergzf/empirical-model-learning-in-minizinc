
int: input_layer_size = 2; % number of features
int: hidden_layer_size = 2;

set of int: xRange = 1..input_layer_size; 
set of int: aRange = 1..hidden_layer_size;
% with bias unit
set of int: xRangeExd = 1..input_layer_size + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

array[xRangeExd] of float: x = [ 1.0, 1.0, 1.0 ]; % add bias unit

array[aRange, xRangeExd] of float: theta1 = [| 1.0, 1.0, 1.0
                                             | 1.0, 1.0, 1.0 |];

array [aRangeExd] of float: theta2 = [ 1.0, 1.0, 1.0 ];

function var float: rectifier(var float: x) = [0, x][1 + (x > 0)]; % array index
% max(0.0, x) % (x > 0) * x

% ------------------------------- forward propagation ---------------------------------

array [aRangeExd] of var float: a;

constraint forall(i in aRangeExd)(
    if i == 1 then a[i] = 1 % bias unit
    else a[i] = rectifier(sum(j in xRangeExd)(x[j] * theta1[i - 1, j]))
    endif
);

var float: h;

constraint h = rectifier(sum(i in aRangeExd)(a[i] * theta2[i]));

solve satisfy;
