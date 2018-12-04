% https://stackoverflow.com/questions/48278879/minizinc-array-of-multidimensional-array/48287027#48287027

int: num_workers = 2;
int: num_weeks = 3;
int: num_days = 4;

array[1..num_workers,1..num_weeks,1..num_days] of int: a =
     array3d(1..num_workers,1..num_weeks,1..num_days,
  [
   % worker 1
   % week1   week2     week3
   1,0,0,1,  1,1,0,1,  0,0,1,1,

   % worker 2
   % week1   week2      week3
   1,0,0,0,  0,0,0,1,   1,1,1,0
  ]
  
);

solve satisfy;

output [show2d(a[2,..,..])];
