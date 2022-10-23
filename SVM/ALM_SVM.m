function [w,b,out] = ALM_SVM(X,y,lam,opts)
%% get size of problem: p is dimension; N is number of data pts
[p,N] = size(X);

%% set parameters
if isfield(opts,'tol')        tol = opts.tol;           else tol = 1e-4;       end
if isfield(opts,'maxit')      maxit = opts.maxit;       else maxit = 500;      end
if isfield(opts,'subtol')     subtol = opts.subtol;     else subtol = 1e-4;    end
if isfield(opts,'maxsubit')   maxsubit = opts.maxsubit; else maxsubit = 5000;  end
if isfield(opts,'w0')         w0 = opts.w0;             else w0 = randn(p,1);  end
if isfield(opts,'b0')         b0 = opts.b0;             else b0 = 0;           end
if isfield(opts,'t0')         t0 = opts.t0;             else t0 = zeros(N,1);  end
if isfield(opts,'beta')       beta = opts.beta;         else beta = 1;         end


alpha0 = 0.5;
alpha = 0.01;
inc_ratio = 2;
dec_ratio = 0.6;

w = w0; b = b0; t = max(0,t0);
% initialize dual variable
u = zeros(N,1);

%% compute the primal residual and save to pres
pres = norm(max(0,1-y.*(X'*w+b)-t));

% save historical primal residual
hist_pres = pres;

%% compute dual residual

dres = sqrt(norm(u'*y)^2+norm(lam*w-X*(y.*u))^2);
hist_dres = dres;

hist_subit = 0;

iter = 0; subit = 0;
%% start of outer loop
while max(pres,dres) > tol & iter < maxit
    iter = iter + 1;
    % call the subroutine to update primal variable (w,b,t)
    w0 = w;
    b0 = b;
    t0 = t;
    
    % fill in the subsolver by yourself
    % if slack variables are introduced, you will have more variables
    [w,b,t] = subsolver(w0,b0,t0,subtol,maxsubit);
    
    hist_subit = [hist_subit; subit];
    
    % update multiplier u
    u = max(0,u+beta*max(0,1-y.*(X'*w+b)-t));
    
    % compute primal residual and save to hist_pres
    pres = norm(max(0,1-y.*(X'*w+b)-t));
    hist_pres = [hist_pres; pres];
    
    % compute gradient of ordinary Lagrangian function about (w,b,t)
    
    grad_w = lam*w-beta*X*(max(0,1-y.*(X'*w+b)-t).*y)-X*(u.*y);
    grad_b = -beta*y'*max(0,1-y.*(X'*w+b)-t)-y'*u;
    grad_t = min(1-beta*max(0,1-y.*(X'*w+b)-t)-u,sign(t).*(1-beta*max(0,1-y.*(X'*w+b)-t)-u));
    
    % compute the dual residual and save to hist_dres
    dres = sqrt(norm(u'*y)^2+norm(lam*w-X*(y.*u))^2);
    hist_dres = [hist_dres; dres];
    
    fprintf('out iter = %d, pres = %5.4e, dres = %5.4e, subit = %d\n',iter,pres,dres,subit);
end

out.hist_pres = hist_pres;
out.hist_dres = hist_dres;
out.hist_subit = hist_subit;

%% =====================================================
% subsolver for primal subproblem
    function [w,b,t] = subsolver(w0,b0,t0,subtol,maxsubit)
        % fill this subsolver
        w = w0;
        b = b0;
        t = t0;
        
        grad_w = lam*w-beta*X*(max(0,1-y.*(X'*w+b)-t).*y)-X*(u.*y);
        grad_b = -beta*y'*max(0,1-y.*(X'*w+b)-t)-y'*u;
        grad_t = min(1-beta*max(0,1-y.*(X'*w+b)-t)-u,sign(t).*(1-beta*max(0,1-y.*(X'*w+b)-t)-u));
        grad_norm = norm([grad_w;grad_b;grad_t]);
        
        iter_inner = 0;
        while grad_norm > subtol & iter_inner < maxsubit
            iter_inner = iter_inner + 1;
            grad_w = lam*w-beta*X*(max(0,1-y.*(X'*w+b)-t).*y)-X*(u.*y);
            grad_b = -beta*y'*max(0,1-y.*(X'*w+b)-t)-y'*u;
            grad_t = min(1-beta*max(0,1-y.*(X'*w+b)-t)-u,sign(t).*(1-beta*max(0,1-y.*(X'*w+b)-t)-u));
            
            w = w-alpha*grad_w;
            b = b-alpha*grad_b;
            t = max(t-alpha*grad_t,0);
            grad_norm = norm([grad_w;grad_b;grad_t]);
        end
        subit = iter_inner;
    end
%=====================================================


end



