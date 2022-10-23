function [x,hist_res] = PG_Lasso(A,b,x0,lam,tol)

% proximal gradient method for solving the Lasso
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


while res > tol
    x0 = x;
       
    % compute gradient at x0
    grad = A'*(A*x0-b);

    % perform one proximal gradient step with stepsize 1/L to get a new x
    x_temp = x0-1/L*grad;

    x = sign(x_temp).*max(0, abs(x_temp)-1/L*lam);
    
    % evaluate norm of the proximal gradient mapping
    res = L * norm(x-x0);
    hist_res = [hist_res; res];
end

end