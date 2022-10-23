function [shrinked_x] =  shrink(x,tau)
    for i=1:numel(x)
        if x(i) >= tau
            shrinked_x(i)=x(i)-tau;
        elseif x(i) <= -tau
            shrinked_x(i)=x(i)+tau;
        else
            shrinked_x(i)=0;
        end
    end
end