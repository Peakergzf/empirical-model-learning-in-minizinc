include "neural_network.mzn";

% Workload Dispatching Problem 
% (variance 1 - WDP_bal)

% ===================================== DATA =====================================

int: n; % the number of jobs
int: m; % the number of cores

constraint assert(n mod m == 0, "n must be a multiple of m.");

% cpi value of each job
array[1..n] of float: cpi;
% (characterize the jobs, low cpi generate more heat)

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
array[1..m] of var float: avgcpi;
constraint forall(k in 1..m)(
    avgcpi[k] = sum(i in 1..n)(cpi[i] * x[i, k]) / m
);

% 2. minimum cpi of the jobs on core k
array[1..m] of var float: mincpi;
constraint forall(k in 1..m)(
    mincpi[k] = min([ max(cpi) - (max(cpi) - cpi[i]) * x[i,k] | i in 1..n ])
);

% 3. average avgcpi of the neighboring cores
array[1..m] of var float: neighcpi;
% TODO 8x6 grid
constraint forall(k in 1..m)(
    neighcpi[k] = sum(j in 1..m where j != k)(avgcpi[j]) / (m - 1)
);

% 4. average avgcpi of all the other cores
array[1..m] of var float: othercpi;
constraint forall(k in 1..m)(
    othercpi[k] = sum(j in 1..m where j != k)(avgcpi[j]) / (m - 1)
);

% ------------------------------------- embed ANN ------------------------------------------

% determine the efficiency of each core using neural network
array[1..m] of var float: eff;
constraint forall(k in 1..m)(
    % input layer size, hidden layer size, (exlude bias unit)
    % input features (include bias unit)
    eff[k] = neural_network( 4, 2, [ avgcpi[k], mincpi[k], neighcpi[k], othercpi[k], 1.0 ] )
);

% ===================================== OBJECTIVE =====================================

% maximize the worst-case core efficiency
var float: obj = min(eff);
solve maximize obj;

% ===================================== OUTPUT =====================================

output [ show(x[i, k] + 0) ++ if k == m then "\n" else " " endif | i in 1..n, k in 1..m ]
       ++
       [ "\nObjective = \(obj)" ];
