function [x, hist_obj, hist_res] = alm_qp(Q,c,A,b,tol,beta,x0)
% augmented Lagrangian method for the quadratic programming
% min_x 0.5*x'*Q*x - c'*x
% s.t.  x >= 0, A*x == b

[m,n] = size(A);

% initialization
v = zeros(m,1);

x = x0;

% compute the residual for the constrain A*x == b
r = A*x-b;

res = norm(r);
grad_err = 1;
hist_res = res;
hist_obj = 0.5*x'*Q*x - c'*x;

% use constant stepsize
alpha = 1/norm(Q + beta*A'*A);

while res > tol | grad_err > tol
    

    % compute the gradient
    grad = Q*x-c + A'*v + beta*A'*(A*x-b);

    % compute violation of optimality condition
    grad_err = 0;
    for i=1:length(x)
        if x(i)==0
            grad_err = grad_err + abs(min(grad(i),0));
        else
            grad_err = grad_err + abs(grad(i));
        end
    end
    while grad_err > tol
        % update x
        x = max(x-alpha*grad, 0);
        % compute the gradient
        
        grad = Q*x-c + A'*v + beta*A'*(A*x-b);
        % compute violation of optimality condition
        grad_err = 0;
        for i=1:length(x)
            if x(i)==0
                grad_err = grad_err + abs(min(grad(i),0));
            else
                grad_err = grad_err + abs(grad(i));
            end
        end
    end
    % compute the residual
    r = A*x-b;
    res = norm(r);
    obj = 0.5*x'*Q*x - c'*x;
    
    % save res and obj
    hist_res = [hist_res; res];
    hist_obj = [hist_obj; obj];
    
    % update multiplier
    v = v + beta*(A*x-b);
end
end