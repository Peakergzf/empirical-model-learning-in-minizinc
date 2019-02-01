
set of int: N(int: k) = 
if k == 1 then { 2, 7 }
elseif k == 2 then { 1, 3, 7, 8 }
elseif k == 3 then { 2, 4, 8, 9 }
elseif k == 4 then { 3, 5, 9, 10 }
elseif k == 5 then { 4, 6, 10, 11 }
elseif k == 6 then { 5, 11, 12 }
elseif k == 7 then { 1, 2, 8, 13 }
elseif k == 8 then { 2, 3, 7, 9, 13, 14 }
elseif k == 9 then { 3, 4, 8, 10, 14, 15 }
elseif k == 10 then { 4, 5, 9, 11, 15, 16 }
elseif k == 11 then { 5, 6, 10, 12, 16, 17 }
elseif k == 12 then { 6, 11, 17, 18 }
elseif k == 13 then { 7, 8, 14, 19 }
elseif k == 14 then { 8, 9, 13, 15, 19, 20 }
elseif k == 15 then { 9, 10, 14, 16, 20, 21 }
elseif k == 16 then { 10, 11, 15, 17, 21, 22 }
elseif k == 17 then { 11, 12, 16, 18, 22, 23 }
elseif k == 18 then { 12, 17, 23, 24 }
elseif k == 19 then { 13, 14, 20, 25 }
elseif k == 20 then { 14, 15, 19, 21, 25, 26 }
elseif k == 21 then { 15, 16, 20, 22, 26, 27 }
elseif k == 22 then { 16, 17, 21, 23, 27, 28 }
elseif k == 23 then { 17, 18, 22, 24, 28, 29 }
elseif k == 24 then { 18, 23, 29, 30 }
elseif k == 25 then { 19, 20, 26, 31 }
elseif k == 26 then { 20, 21, 25, 27, 31, 32 }
elseif k == 27 then { 21, 22, 26, 28, 32, 33 }
elseif k == 28 then { 22, 23, 27, 29, 33, 34 }
elseif k == 29 then { 23, 24, 28, 30, 34, 35 }
elseif k == 30 then { 24, 29, 35, 36 }
elseif k == 31 then { 25, 26, 32, 37 }
elseif k == 32 then { 26, 27, 31, 33, 37, 38 }
elseif k == 33 then { 27, 28, 32, 34, 38, 39 }
elseif k == 34 then { 28, 29, 33, 35, 39, 40 }
elseif k == 35 then { 29, 30, 34, 36, 40, 41 }
elseif k == 36 then { 30, 35, 41, 42 }
elseif k == 37 then { 31, 32, 38, 43 }
elseif k == 38 then { 32, 33, 37, 39, 43, 44 }
elseif k == 39 then { 33, 34, 38, 40, 44, 45 }
elseif k == 40 then { 34, 35, 39, 41, 45, 46 }
elseif k == 41 then { 35, 36, 40, 42, 46, 47 }
elseif k == 42 then { 36, 41, 47, 48 }
elseif k == 43 then { 37, 38, 44 }
elseif k == 44 then { 38, 39, 43, 45 }
elseif k == 45 then { 39, 40, 44, 46 }
elseif k == 46 then { 40, 41, 45, 47 }
elseif k == 47 then { 41, 42, 46, 48 }
else { 42, 47 }
endif;