function [x, hist_res] = quadMin_gd(A,b,x0,tol)

% steepest gradient method for solving
% min_x 0.5*x'*A*x - b'*x

% get the size of the problem
n = length(b);

x = x0;

% compute gradient of the objective
grad = A*x-b;

% evaluate the norm of gradient
res = norm(grad);

% save the value of res
hist_res = res;

while res > tol
    % compute the stepsize alpha by exact line search
    alpha = res^2/(grad'*A*grad);
    
    % update x
    x = x-alpha*grad;
    
    % compute gradient of the objective
    grad = A*x-b;
    
    % evaluate the norm of gradient
    res = norm(grad);
    
    % save the value of res
    hist_res = [hist_res; res];
end

end

