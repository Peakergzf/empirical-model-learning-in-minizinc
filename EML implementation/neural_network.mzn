
% assume one hidden layer?
% confusing naming? n, x

function var float: neural_network(int: n, int: hidden_layer_size, array[int] of var float: x) = 0.0


% set of int: xRange = 1..n; 
% set of int: aRange = 1..hidden_layer_size;
% % with bias unit
% set of int: xRangeExd = 1..n + 1; 
% set of int: aRangeExd = 1..hidden_layer_size + 1;

% % TODO

% % array[aRange, xRangeExd] of float: theta1 = [| 1.0, 1.0, 1.0
% %                                              | 1.0, 1.0, 1.0 |];

% % array [aRangeExd] of float: theta2 = [ 1.0, 1.0, 1.0 ];


% % ------------------------------- forward propagation ---------------------------------

% array [aRangeExd] of var float: a;

% constraint forall(i in aRangeExd)(
%     if i == 1 then a[i] = 1 % bias unit
%     else a[i] = tanh(sum(j in xRangeExd)(x[j] * theta1[i - 1, j]))
%     endif
% );

% var float: h;

% constraint h = tanh(sum(i in aRangeExd)(a[i] * theta2[i]));

