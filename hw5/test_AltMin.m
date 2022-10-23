n = 100;
randn('seed',20220315);
[Q,~] = qr(randn(n));
lam_min = 1;
lam_max = 100;
lam = linspace(lam_min,lam_max,n);

A = Q*diag(lam)*Q';
A = (A + A')/2;
b = randn(n,1);

x0 = randn(n,1);
lb = -ones(n,1);
ub = ones(n,1);

maxit = 200;

%% compute optimal solution by MATLAB built-in function quadprog
[xopt, fopt] = quadprog(A, -b, [],[],[],[],lb,ub);

%% call your AltMin method
t0 = tic;
[x_s, hist_obj_s] = quadMin_AltMin(A,b,x0,maxit,lb,ub);
t1 = toc(t0);

fprintf('Student code: Total running time is %5.4f\n', t1);

fprintf('Final objective value is %5.4f\n', .5*x_s'*A*x_s - b'*x_s);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_obj_s - fopt, 'b-','linewidth',2);

xlabel('Iteration number','fontsize',12);
ylabel('objective error','fontsize',12);
title('Student AltMin solver')
print(fig,'-dpdf','student_AltMin_result');

%% call instructor's AltMin method
t0 = tic;
[x_p, hist_obj_p] = quadMin_AltMin_p(A,b,x0,maxit,lb,ub);
t1 = toc(t0);

fprintf('Intructor code: Total running time is %5.4f\n', t1);

fprintf('Final objective value is %5.4f\n', .5*x_p'*A*x_p - b'*x_p);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_obj_p - fopt, 'b-','linewidth',2);

xlabel('Iteration number','fontsize',12);
ylabel('objective error','fontsize',12);
title('Instructor AltMin solver')
print(fig,'-dpdf','instructor_AltMin_result');

%% call instructor's PG method
t0 = tic;
[x_p, hist_obj_p] = quadMin_pg_p(A,b,x0,maxit,lb,ub);
t1 = toc(t0);

fprintf('Intructor code: Total running time is %5.4f\n', t1);

fprintf('Final objective value is %5.4f\n', .5*x_p'*A*x_p - b'*x_p);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_obj_p - fopt, 'b-','linewidth',2);

xlabel('Iteration number','fontsize',12);
ylabel('objective error','fontsize',12);
title('Instructor PG solver')
print(fig,'-dpdf','instructor_PG_result');