/*
k-w-1, k-w, k-w+1
k-1,        k+1
k+w-1, k+w, k+w+1
*/

set of int: N(int: k, int: m, int: h) = let {
  int: w = m div h;
  int: row = (k-1) div w;
  int: col = (k-1) mod w;
  set of int: currow = if col = 0 then { k+1 } 
                       elseif col = w-1 then { k-1 } 
                       else { k-1, k+1} endif;
  set of int: prevrow = if row = 0 then {} 
                        elseif col = 0 then { k-w, k-w+1 } 
                        elseif col = w-1 then { k-w, k-w-1 } 
                        else { k-w-1, k-w, k-w+1 } endif;  
  set of int: nextrow = if row = h-1 then {} 
                        elseif col = 0 then { k+w, k+w+1 } 
                        elseif col = w-1 then { k+w, k+w-1 } 
                        else { k+w-1, k+w, k+w+1 } endif;
} in currow union prevrow union nextrow;
