
array [int, int] of int: b = [| 1,2,3, |
                                4,5,6  |];


array [int, int, int] of int: a = [| |1,2,3,| 4,5,6|,
                                     |7,8,9,|10,11,12| |];
solve satisfy;

output [show2d(a[1,..,..]) ++ show2d(b[..,..])];