m = 200;
n = 100;
xorg = randn(n,1);
A = randn(m,n);
for i = 1:m
    A(i,:) = A(i,:)/norm(A(i,:));
end

b = A*xorg;
s = 10;
id = randsample(m,s);
b(id) = b(id) + randn(s,1);

x0 = randn(n,1);

%% compute the optimal value by linear programming
Aineq = [zeros(m,n), eye(m), -eye(m);zeros(m,n), -eye(m), -eye(m)];
bineq = zeros(2*m,1);
Aeq = [A, -eye(m), zeros(m)];
beq = b;
f = [zeros(n+m,1); ones(m,1)];
[w,fopt] = linprog(f,Aineq,bineq,Aeq,beq);

xopt = w(1:n);

%% call student's ADMM solver
rho = 1;
tol = 1e-6;
maxit = 500;
t0 = tic;
[x_s, hist_obj_s, out_s] = ADMM_LAD(A,b,rho,tol,maxit,x0);
t1 = toc(t0);

fprintf('Student solver: Total running time of instructor code is %5.4f\n', t1);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_obj_s - fopt, 'r-','linewidth',2);

set(gca,'fontsize',12);

xlabel('iteration number','fontsize',12);
ylabel('objective error','fontsize',12);

title('Student code','fontsize',12);

print(fig, '-dpdf','student_ADMM_results')


%% call instructor's ADMM solver
rho = 1;
tol = 1e-6;
maxit = 500;
t0 = tic;
[x_p, hist_obj_p, out_p] = ADMM_LAD_p(A,b,rho,tol,maxit,x0);
t1 = toc(t0);

fprintf('Instructor solver: Total running time of instructor code is %5.4f\n', t1);

fig = figure('papersize',[5,4],'paperposition',[0,0,5,4]);

semilogy(hist_obj_p - fopt, 'b-','linewidth',2);

set(gca,'fontsize',12);

xlabel('iteration number','fontsize',12);
ylabel('objective error','fontsize',12);

title('Instructor code','fontsize',12);

print(fig, '-dpdf','instructor_ADMM_results')

