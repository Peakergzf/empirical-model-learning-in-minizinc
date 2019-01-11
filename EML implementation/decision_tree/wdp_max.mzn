include "decision_tree.mzn";
include "neigh_k.mzn";

% Workload Dispatching Problem 
% (variance 2 - WDP_max)

% ===================================== DATA =====================================

int: n; % the number of jobs
int: m; % the number of cores

int: h; % the height of the grid

constraint assert(n mod m == 0, "n must be a multiple of m.");

% cpi value of each job
array[1..n] of float: cpi;

% ---------------------------------- decision tree ----------------------------------

array[1..m, int] of int: feature_idx;
array[1..m, int] of REL: feature_rel;
array[1..m, int] of float: feature_val;

array[1..m, int] of int: child;

array[1..m, int] of int: val; 

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

% ------------------------------------- input features for DT ------------------------------

% 1. average cpi of the jobs on core k
array[1..m] of var float: self_cpi_mean = [ sum(i in 1..n)(cpi[i] * x[i, k]) / m | k in 1..m ];

% 2. minimum cpi of the jobs on core k
array[1..m] of var 0.0..max(cpi): self_cpi_min = [ min([ max(cpi) - (max(cpi) - cpi[i]) * x[i, k] | i in 1..n ]) | k in 1..m ];


% (set of neighbors for each core)
array[1..m] of set of int: neigh = [ N(k, m, h) | k in 1..m ];

% 3. average avgcpi of all the other cores
array[1..m] of var float: all_cpi_mean = [ 
    sum(j in 1..m diff neigh[k] diff {k} ) (self_cpi_mean[j]) / (m - 1 - card(neigh[k])) 
| k in 1..m ];

% 4. average avgcpi of the neighboring cores
array[1..m] of var float: neigh_cpi_mean = [ 
    sum(j in neigh[k]) (self_cpi_mean[j]) / card(neigh[k]) 
| k in 1..m ];

% ------------------------------------- embed DT ------------------------------------------

% determine if the efficiency of each core is larger than a certain threshold using decision tree
array[1..m] of var 0..1: eff = [ 
             decision_tree( 
                            [ self_cpi_min[k], self_cpi_mean[k], all_cpi_mean[k], neigh_cpi_mean[k] ],
                            feature_idx[k, ..], 
                            array1d(feature_rel[k, ..]), 
                            feature_val[k, ..],
                            child[k, ..],
                            val[k, ..] ) | k in 1..m ];

% ===================================== OBJECTIVE =====================================

% maximize the number of cores having an efficiency larger than a certain threshold
var int: obj;

constraint obj = sum(k in 1..m)(eff[k]);

solve maximize obj;

% ===================================== OUTPUT =====================================

% output [ show(x[i, k] + 0) ++ if k == m then "\n" else " " endif | i in 1..n, k in 1..m ];

output  
      ["\njob -> core mapping:\n"]
       ++
       [ if k == 1 then "\(i)->" else "" endif ++
         if fix(x[i, k]) then "\(k), " else "" endif
%          ++if i mod ((n div m)*2) == 0  /\ k == m then "\n" else "" endif
         | i in 1..n, k in 1..m ]
       ++
       ["\naverage CPI map:\n\t"]
       ++
       [ show_float(6, 5, self_cpi_mean[k]) ++ if k mod (m div h) == 0 then "\n\t" else  ", " endif | k in 1..m ]
       ++
       [ "\nefficiency satisfaction map:\n\t" ]
       ++
       [ show(eff[k]) ++ if k mod (m div h) == 0 then "\n\t" else  ", " endif | k in 1..m ]
       ++
       [ "\nnumber of satisfied efficiency constraints: \(obj)\n" ];
