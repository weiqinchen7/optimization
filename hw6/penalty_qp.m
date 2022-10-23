function [x, hist_obj, hist_res] = penalty_qp(Q,c,A,b,tol,mu0,mu1,x0)
% quadratic penalty method for the quadratic programming
% min_x 0.5*x'*Q*x - c'*x
% s.t.  x >= 0, A*x == b

mu = mu0;
x = x0;

% compute the residual for the constraint A*x == b
r = A*x-b;

res = norm(r);
grad_err = 1;
hist_res = res;
hist_obj = 0.5*x'*Q*x - c'*x;

while (res > tol | grad_err > tol) & mu < mu1
    % use constant stepsize
    alpha = 1/norm(Q + mu*A'*A);
    % compute the gradient
    grad = Q*x-c+mu*A'*(A*x-b);

    % compute violation of optimality condition
    grad_err = 0;
    for i=1:length(x)
        if x(i)==0
            grad_err = grad_err+abs(min(grad(i),0));
        else
            grad_err = grad_err+abs(grad(i));
        end
    end
    while grad_err > tol
        % update x
        x = max(x-alpha*grad,0);
        % compute the gradient
        
        grad = Q*x-c+mu*A'*(A*x-b);
        % compute violation of optimality condition
        grad_err = 0;
        for i=1:length(x)
            if x(i)==0
                grad_err = grad_err+abs(min(grad(i),0));
            else
                grad_err = grad_err+abs(grad(i));
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
    
    % increase the penalty parameter
    mu = 5*mu;
end
end