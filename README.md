# Empirical Model Learning in MiniZinc

### Overview

This repository contains the [MiniZinc](https://www.minizinc.org/) implementation of the two variances of the thermal-aware workload dispatching (WDP) case study from the paper [Empirical decision model learning](https://www.sciencedirect.com/science/article/abs/pii/S0004370216000126). 

The case study consist in mapping a set of jobs to a multi-core CPU so as to maximize the core efficiencies. 

In the first variance (WDP<sub>bal</sub>), neural networks are used to predict the efficiency of each core, the objective is to maximize the worst-case core efficiency.

In the second variance (WDP<sub>max</sub>), decision trees are used to predict the efficiency of each core, the objective is to maximize the number of cores having an efficiency larger than a certain threshold.


### Folder structure

`wdp-bal-neural-network` and `wdp-max-decition-tree` contain the implementation of the two variances described above.

In each folder, `data-preprocessing` contains the code that prepares for the data used in the MiniZinc model; 

`result-verification` contains the model that verifies the result mentioned in the paper (i.e. the solutions are added as constraints to the model to compute the objective values in order to compare with the paper).

`handwritten-digits` contains an alternative example of embedding neural network in an optimisation model. <br>

### TODO

 - [ ] finalise decision tree representation (left branching is always LT, no need for other enum)
 - [ ] float `in` `var-dom`
 - [ ] [adversarial machine learning](https://en.wikipedia.org/wiki/Adversarial_machine_learning)