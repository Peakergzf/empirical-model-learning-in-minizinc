
int: n = 13; % the number of nodes

int: NO_CHILD = 0;

set of int: NODE = 1..n;

% attributes
var int: A;
var int: B;
var bool: C; % true for high, false for low

% left branching condition for each node
array[NODE] of var bool: cond = [
%                             1
%                      /            \
                  A <= 20,    
%                    /                 \
%                    2                  3
%                  /  \               /    \
            B <= 10,             not C,  
%               /       \            /       \   
%             4          5           6        7
%          /      \      |          /  \      |
    not C,             true,  B <= 20,       true, 
%       /           \             /      \
%      8             9           10       11
%      |            / \           |       |
    true,    A <= 10,             true,   true, 
%                 /      \
%                12      13
%                |        |
                true,    true]; 

% breadth first ordering of the left child for each non-laef node (0 for leaf node)
array[NODE] of int: child = [2, 4, 6, 8, 0, 10, 0, 0, 12, 0, 0, 0, 0];
% index                      1  2  3  4  5  6   7  8  9   10 11 12 13

% output value(class) for leaf node (-1 for non-leaf node)
array[NODE] of int: val = [-1, -1, -1, -1, 0, -1, 1, 1, -1, 0, 1, 0, 1]; 
% index                     1   2   3   4  5   6  7  8   9  10 11 12 13

function var int: res(var int: i) = 
    if child[i] == NO_CHILD then val[i]
    elseif cond[i] then res(child[i])
    else res(child[i + 1])
    endif;

constraint res(1) == 1;

solve satisfy;
