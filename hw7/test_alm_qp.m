clear; close all;
m = 20;
n = 50;
A = [eye(m), randn(m,n-m)];
b = rand(m,1) + 0.1;
Q = randn(n);
Q = Q'*Q;
Q = Q/max(abs(Q(:)));
c = randn(n,1);

x0 = max(0,randn(n,1));
tol = 1e-6;
beta = 0.5;

%% call quadprog to get optimal solution
[xopt, fopt] = quadprog(Q, -c, [],[], A,b,zeros(n,1),[]);

%% call student's solver
t0 = tic;
[x_s, hist_obj_s, hist_res_s] = alm_qp(Q,c,A,b,tol,beta,x0);
t1 = toc(t0);

fprintf('Student solver: Total running time of instructor code is %5.4f\n', t1);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(abs(hist_obj_s - fopt), 'b-','linewidth',2);
hold on
semilogy(hist_res_s,'r-','linewidth',2);
legend('Objective error','Feasibility violation')

set(gca,'fontsize',12);

xlabel('outer iteration number','fontsize',12);
ylabel('error','fontsize',12);

title('Student code','fontsize',12);

print(fig, '-dpdf','student_ALM_results')

%% call instructor's solver
t0 = tic;
[x_p, hist_obj_p, hist_res_p] = alm_qp_p(Q,c,A,b,tol,beta,x0);
t1 = toc(t0);

fprintf('Instructor solver: Total running time of instructor code is %5.4f\n', t1);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(abs(hist_obj_p - fopt), 'b-','linewidth',2);
hold on
semilogy(hist_res_p,'r-','linewidth',2);
legend('Objective error','Feasibility violation')

set(gca,'fontsize',12);

xlabel('outer iteration number','fontsize',12);
ylabel('error','fontsize',12);

title('Instructor code','fontsize',12);

print(fig, '-dpdf','instructor_ALM_results')
