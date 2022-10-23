function [x,hist_res] = APG_Lasso(A,b,x0,lam,tol)

% accelerated proximal gradient method for solving the Lasso
% min_x .5*||A*x-b||^2 + lam*||x||_1

% compute Lipschitz constant
L = norm(A*A');

% compute gradient at x0
grad = A'*(A*x0-b);

% perform one proximal gradient step with stepsize 1/L to get a new x
x_temp = x0-1/L*grad;
x = sign(x_temp).*max(0, abs(x_temp)-1/L*lam);

% evaluate norm of the proximal gradient mapping
res = L * norm(x-x0);

hist_res = res;

x0 = x;

% y used to denote the extrapolated point

y = x; 

% used to compute the extrapolation weight
t0 = 1;


while res > tol
    
    % compute a gradiet at y
    grad = A'*(A*y-b);;
    
    % perform one proximal gradient step with stepsize 1/L to get a new x
    y_temp = y-1/L*grad;
    x = sign(y_temp).*max(0, abs(y_temp)-1/L*lam);
    
    % update t value to compute extrapolation weight
    
    t1 = (1+sqrt(1+4*t0^2))/2;
    
    % update the extrapolated point y
    
    y = x+(t0-1)/t1*(x-x0);
    
    % evaluate norm of the proximal gradient mapping
    res = L * norm(x-x0);
    hist_res = [hist_res; res];
    
    t0 = t1; x0 = x;
end

end