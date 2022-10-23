n = 100;
[Q,~] = qr(randn(n));
lam_min = 1;
lam_max = 100;
lam = linspace(lam_min,lam_max,n);

A = Q*diag(lam)*Q';
b = randn(n,1);

tol = 1e-8;
x0 = randn(n,1);

%% call your DFP solver
t0 = tic;
[x_s, hist_res_s] = quadMin_DFP(A,b,x0,tol);
t1 = toc(t0);

fprintf('Student solver: Total running time is %5.4f\n', t1);

fprintf('Final objective value is %5.4f\n', .5*x_s'*A*x_s - b'*x_s);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_res_s, 'b-','linewidth',2);

xlabel('Iteration number','fontsize',12);
ylabel('Gradient norm','fontsize',12);
title('Student solver')
print(fig,'-dpdf','student_DFP_result');

%% call instructor's method

t0 = tic;

[x_p, hist_res_p] = quadMin_DFP_p(A,b,x0,tol);

t1 = toc(t0);

fprintf('Instructor solver: Total running time is %5.4f\n', t1);

fprintf('Final objective value is %5.4f\n', .5*x_p'*A*x_p - b'*x_p);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_res_p, 'k-','linewidth',2);

xlabel('Iteration number','fontsize',12);
ylabel('Gradient norm','fontsize',12);
title('Instructor solver')
print(fig,'-dpdf','instructor_DFP_result');
