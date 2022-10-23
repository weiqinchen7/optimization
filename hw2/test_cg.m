clear; close all;

%% generate the data
n = 100;
e = ones(n,1);
A = spdiags([-2*e 4.5*e -2*e], -1:1, n,n);
b = rand(n,1) - 1;
x0 = zeros(n,1);
tol = 1e-5;

%% call your solver and show the results

t0 = tic;
[x_s, hist_res_s] = quadMin_cg(A,b,x0,tol);

t1 = toc(t0);

% print results

fprintf('Results by student code\n');

fprintf('Total running time is %5.4f\n', t1);

fprintf('Final objective value is %5.4f\n', .5*x_s'*A*x_s - b'*x_s);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_res_s, 'b-','linewidth',2);

xlabel('Iteration number','fontsize',12);
ylabel('Gradient norm','fontsize',12);
title('Student CG method', 'fontsize', 12);

%% call instructor's solver

t0 = tic;
[x_p, hist_res_p] = quadMin_cg_p(A,b,x0,tol);

t1 = toc(t0);

% print results

fprintf('Results by Instructor code\n');

fprintf('Total running time is %5.4f\n', t1);

fprintf('Final objective value is %5.4f\n', .5*x_p'*A*x_p - b'*x_p);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_res_p, 'k-','linewidth',2);

xlabel('Iteration number','fontsize',12);
ylabel('Gradient norm','fontsize',12);
title('Instructor CG method', 'fontsize', 12);