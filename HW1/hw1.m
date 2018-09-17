% Task 1
mean = 0;
std = 5;
arr = std.*randn(1000000, 1) + mean;
disp('Test 1 passes');

% Task 2
arr_size = size(arr);
tic % begin timer
for i = 1:arr_size
    arr(i, 1) = arr(i, 1) + 1;
end
disp('Test 2 passes')
toc % end timer

% Task 3
tic % begin timer
arr = arr + ones(1000000, 1);
disp('Test 3 passes');
elapsedTime = toc; % end timer
disp('Elapsed time with disp (seconds): ')
disp(elapsedTime)
fprintf('Elapsed time with fprintf: %.4f seconds\n', elapsedTime);

% Task 4
x = [2:2:99].';
y = 2.^x;
plot(x, y)
disp('Test 4 passes');

% Task 5
A = ones(10, 10);
B = reshape(0:99, 10, 10);
C = A + B;
assert(all(C(:) == (1:100)') == 1);
disp('Test 5 passes');

% Task 6
A = rand(5, 3);
B = rand(3, 5);
C = A*B;
newC = zeros(5, 5);
for row = 1:5
    for col = 1:5
        dotProduct = 0;
        for i = 1:3
            dotProduct = dotProduct + A(row, i).*B(i, col);
        end
        newC(row, col) = dotProduct;
    end
end
tol = 0.00001;
assert(all((C(:)-newC(:)) < tol) == 1); % Ensure C and newC are equal; have to use this check because of rounding/precision errors
disp('Test 6 passes');

% Task 7
a = [1 2 3];
b = [2 3 4];
dot1 = dot(a, b);
dot2 = a*b.';
dot3 = sum(a.*b);
if dot1 == dot2 & dot2 == dot3
    disp("Test 7 passes");
end

% Task 8
x1 = [0.5 0 1.5];
x2 = [1 1 0];
x1L1norm = sum(abs(x1));
x1L2norm = sqrt(sum(x1.^2));
x1L1normActual = norm(x1, 1);
x1L2normActual = norm(x1, 2);
x2L1norm = sum(abs(x2));
x2L2norm = sqrt(sum(x2.^2));
x2L1normActual = norm(x2, 1);
x2L2normActual = norm(x2, 2);
assert(x1L1norm == x1L1normActual);
assert(x1L2norm == x1L2normActual);
assert(x2L1norm == x2L1normActual);
assert(x2L2norm == x2L2normActual);
disp('Test 8 passes');

% Task 9
syms x y z
eqn1 = 2.*x + y + 3.*z == 1;
eqn2 = 2.*x + 6.*y + 8.*z == 3;
eqn3 = 6.*x + 8.*y + 18.*z == 5;
[A,B] = equationsToMatrix([eqn1, eqn2, eqn3], [x, y, z]);
X = sprintf('x = %s, y = %s, z = %s', linsolve(A,B));
fprintf('Test 9 passes with solution: %s\n', X);

% Task 10
A = rand(2, 2);
[B] = normalize_rows(A);
disp('Test 10 passes');
disp('A = ')
disp(A)
disp('B = ')
disp(B)

% Task 11
a = fib(1);
b = fib(2);
c = fib(3);
d = fib(5);
e = fib(10);
assert(a == 1);
assert(b == 1);
assert(c == 2);
assert(d == 5);
assert(e == 55);
disp('Test 11 passes');

% Task 12
% Representation: [red_true, green_true, blue_true, circle_true, triangle_true, square_true]
% red_true = 0 or 1, whether the object is red
% green_true = 0 or 1, whether the object is green
% blue_true = 0 or 1, whether the object is blue
% circle_true = 0 or 1, whether the object is a circle
% triangle_true = 0 or 1, whether the object is a triangle
% square_true = 0 or 1, whether the object is a square
red_circle = [1 0 0 1 0 0];
blue_triangle = [0 0 1 0 1 0];
blue_circle = [0 0 1 1 0 0];
green_triangle = [0 1 0 0 1 0];
red_square = [1 0 0 0 0 1];
disp('Test 12 passes');