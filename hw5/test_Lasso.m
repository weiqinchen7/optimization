clear; close all;
%% generate data
randn('seed',20220315);
rand('seed',20220315);
m = 200;
n = 2000;
s = 20;

% generate a sensing matrix
A = randn(m,n);
for i = 1:m
    A(i,:) = A(i,:)/norm(A(i,:));
end

% generate a sparse vector
xorg = zeros(n,1);
xorg(randsample(n,s)) = randn(s,1);

% obtain the measurement
b = A*xorg;

tol = 1e-4;

%% run student's proximal gradient method
x0 = zeros(n,1);
lam = 1e-3;

L = norm(A*A');
time0 = tic;
[x_pg, hist_res_pg] = PG_Lasso(A,b,x0,lam,tol);
time_pg = toc(time0);

%% run student's accelerated proximal gradient method
time0 = tic;
[x_apg, hist_res_apg] = APG_Lasso(A,b,x0,lam,tol);
time_apg = toc(time0);

%% print results by student solver

fprintf('Results by student solver\n')

fprintf('PG method: time = %5.4f, Relative Error = %5.4e\n', time_pg, norm(x_pg-xorg) / norm(xorg) );
fprintf('APG method: time = %5.4f, Relative Error = %5.4e\n', time_apg, norm(x_apg-xorg) / norm(xorg) );

close all;

fig = figure('papersize',[4,5],'paperposition',[0,0,4,5]);

subplot(2,1,1);

plot(xorg, 'ro','markersize',8);
hold on
plot(x_pg, 'b*','markersize',8);

title('PG method');

subplot(2,1,2);
plot(xorg, 'ro','markersize',8);
hold on
plot(x_apg, 'k+','markersize',8);
title('APG method');
print(fig,'-dpdf','student_Lasso_result');


%% run instructor's proximal gradient method
x0 = zeros(n,1);
lam = 1e-3;

L = norm(A*A');
time0 = tic;
[x_pg, hist_res_pg] = PG_Lasso_p(A,b,x0,lam,tol);
time_pg = toc(time0);

%% run instructor's accelerated proximal gradient method
time0 = tic;
[x_apg, hist_res_apg] = APG_Lasso_p(A,b,x0,lam,tol);
time_apg = toc(time0);

%% print results by instructor solver

fprintf('Results by Instructor solver\n')

fprintf('PG method: time = %5.4f, Relative Error = %5.4e\n', time_pg, norm(x_pg-xorg) / norm(xorg) );
fprintf('APG method: time = %5.4f, Relative Error = %5.4e\n', time_apg, norm(x_apg-xorg) / norm(xorg) );

close all;

fig = figure('papersize',[4,5],'paperposition',[0,0,4,5]);

subplot(2,1,1);

plot(xorg, 'ro','markersize',8);
hold on
plot(x_pg, 'b*','markersize',8);

title('PG method');

subplot(2,1,2);
plot(xorg, 'ro','markersize',8);
hold on
plot(x_apg, 'k+','markersize',8);
title('APG method');
print(fig,'-dpdf','instructor_Lasso_result');

