function [x, hist_res] = quadMin_BB(A,b,x0,tol)

% BB method for solving
% min_x 0.5*x'*A*x - b'*x

x = x0;

%% perform one steepest gradient descent with exact line search

% compute gradient of the objective
grad = A * x - b ;

% evaluate the norm of gradient
res = norm(grad);

% save the value of res
hist_res = res;

alpha = res^2 / (grad' * A * grad);
    
% update x by steepest gradient descent
x = x - alpha * grad;

%% main iteration
while res > tol

    % compute s and y
    s = x - x0;
    x0 = x;
    
    % save the old grad
    grad0 = grad;
    
    % compute a new grad
    grad = A * x - b;
    
    y = grad - grad0;
    
    % compute alpha by option I or option II
    
    alpha = norm(s)^2 / dot(s, y);
    
    % update x
    x = x - alpha * grad;
    
    % evaluate the norm of gradient
    res = norm(grad);
    
    % save the value of res
    hist_res = [hist_res; res];
end

end

