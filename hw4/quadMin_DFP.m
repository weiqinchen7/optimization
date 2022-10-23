function [x, hist_res] = quadMin_DFP(A,b,x0,tol)

% DFP method for solving
% min_x 0.5*x'*A*x - b'*x

x = x0;

%% perform one steepest gradient descent with exact line search

% compute gradient of the objective
grad = A*x - b;

% evaluate the norm of gradient
res = norm(grad);

% save the value of res
hist_res = res;

% choose the initial H matrix

H = eye(length(b));

%% main iteration
while res > tol
    
    % compute the search direction
    p = -H*grad;
    
    % use exact line search
    alpha = norm(p)^2 / ( p'*(A*p) );
    
    % update x 
    x = x + alpha*p;
    
    grad0 = grad;
    % compute gradient of the objective
    grad = A*x - b;
    
    % compute s and y
    s = alpha*p;
    
    y = grad - grad0;
    
    % update H matrix
    H = H - (H*y*y'*H)/(y'*H*y) + (s*s')/(s'*y);  
    
    % evaluate the norm of gradient
    res = norm(grad);
    
    % save the value of res
    hist_res = [hist_res; res];
end

end