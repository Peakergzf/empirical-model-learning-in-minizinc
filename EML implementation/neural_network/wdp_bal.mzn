include "neural_network.mzn";
include "neigh_k.mzn";

% Workload Dispatching Problem 
% (variance 1 - WDP_bal)

% ===================================== DATA =====================================

int: n; % the number of jobs
int: m; % the number of cores

int: h;

constraint assert(n mod m == 0, "n must be a multiple of m.");

% cpi value of each job
array[1..n] of float: cpi;

% -------------------------------- data for neural network ----------------------------

int: input_layer_size;
int: hidden_layer_size;

set of int: xRange = 1..input_layer_size; 
set of int: aRange = 1..hidden_layer_size;
% with bias unit
set of int: xRangeExd = 1..input_layer_size + 1; 
set of int: aRangeExd = 1..hidden_layer_size + 1;

% assume one hidden layer, one output unit

array[1..m, aRange, xRangeExd] of float: theta1;

array[1..m, aRangeExd] of float: theta2;

% ===================================== DECISION =====================================

% x[i, k] = true  iff job i is mapped on core k and false otherwise
array[1..n, 1..m] of var bool: x;

% ===================================== CONSTRSINTS =====================================

% ------------------------------------- verification -------------------------------------

% each job is mapped on a single core
constraint forall(i in 1..n)(
    sum(k in 1..m)(x[i, k]) = 1
);

% the same number of jobs run on each core
constraint forall(k in 1..m)(
    sum(i in 1..n)(x[i, k]) = n div m 
);

% ------------------------------------- input features for ANN ------------------------------

% 1. average cpi of the jobs on core k
array[1..m] of var float: avgcpi = [ sum(i in 1..n)(cpi[i] * x[i, k]) / m | k in 1..m ];

% 2. minimum cpi of the jobs on core k
array[1..m] of var 0.0..max(cpi): mincpi = [ min([ max(cpi) - (max(cpi) - cpi[i]) * x[i, k] | i in 1..n ]) | k in 1..m ];


% (set of neighbors for each core)
array[1..m] of set of int: neigh = [ N(k, m, h) | k in 1..m ];


% 3. average avgcpi of the neighboring cores
array[1..m] of var float: neighcpi = [ 
    sum(j in neigh[k]) (avgcpi[j]) / card(neigh[k]) 
| k in 1..m ];

% 4. average avgcpi of all the other cores
array[1..m] of var float: othercpi = [
    sum(j in 1..m diff neigh[k] diff {k} ) (avgcpi[j]) / (m - 1 - card(neigh[k])) 
| k in 1..m ];



% ------------------------------------- embed ANN ------------------------------------------

% determine the efficiency of each core using neural network
    % input layer size, hidden layer size, (exclude bias unit)
    % input features (include bias unit)
array[1..m] of var float: eff = [ 
             neural_network( input_layer_size, 
                             hidden_layer_size, 
                             [ avgcpi[k], mincpi[k], neighcpi[k], othercpi[k], 1.0 ],
                             theta1[k, .., ..],
                             theta2[k, ..] ) | k in 1..m ];

% ===================================== OBJECTIVE =====================================

% maximize the worst-case core efficiency
var 0.0..1.0: obj;
constraint forall (i in index_set(eff)) (obj <= eff[i]);
solve maximize obj;

% ===================================== OUTPUT =====================================

output [ show(x[i, k] + 0) ++ if k == m then "\n" else " " endif | i in 1..n, k in 1..m ]
       ++
       [ "\nObjective = \(obj)" ];
