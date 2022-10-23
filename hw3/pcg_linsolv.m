function y = pcg_linsolv(C, r)
z = Linsolv_1(C', r);
y = Linsolv_2(C, z);
end


function z = Linsolv_1(C, r)
num = length(C); 
z = ones(1,num);
for i = 1 : num
    s = 0;
    for j = 1 : i - 1
        s = s + C(i,j) * z(j);
    end
    z(i) = (r(i) - s) / C(i,i);
end
z = z'; 
end

function y = Linsolv_2(C, z)
num = length(C);
y = ones(1,num);
for i = num : -1 : 1
    s = 0;
    for j = num : -1 : i + 1
        s = s + C(i, j) * y(j);
    end
    y(i) = (z(i) - s) / C(i,i);
end
y = y';
end