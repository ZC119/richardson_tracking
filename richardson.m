k = 1000;

% 矩阵
A = randn(k);
A = A'*A;

% 待求解未知数
x = randn(k, 1)

b = A * x;

C = norm(A, inf);

w = 2 / C;

eps = 1e-3;

x_iter = zeros(k, 1);
x_iter_old = x;

n = 1;

tic()

while norm(x_iter-x_iter_old) > eps
   x_iter_old = x_iter;
   x_iter = x_iter + w * (b - A * x_iter); 
   n = n+1;
end


x_iter

n

toc()
