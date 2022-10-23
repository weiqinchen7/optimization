gradient_all = [];
x1 = -1.5;
x2 = 1.5;
for i = 1:10
    gradient = [-200*x1*(x2-x1^2)+2*x1-2; 100*(x2-x1^2)];
    hessian = [-200*(x2-3*x1^2)+2 -200*x1; -200*x1 100];
    p= -inv(hessian)*gradient;
    gradient_all = [gradient_all; norm(gradient)];
    x1=x1+p(1);
    x2=x2+p(2);
end

close all;
figure;
x=1:10;
plot(x,gradient_all, 'r')
title('Newton Method')
xlabel('Iteration number','fontsize',12);
ylabel('Gradient norm','fontsize',12);