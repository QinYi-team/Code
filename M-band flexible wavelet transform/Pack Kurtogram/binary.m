function a = binary(i,k)
% return the coefficients of the binary expansion of i:i的二的指数表达式系数
% i = a(1)*2^(k-1) + a(2)*2^(k-2) + ... + a(k)

if i>=2^k
   error('i must be such that i < 2^k !!')
end
a = zeros(1,k);
temp = i;
for l = k-1:-1:0
   a(k-l) = fix(temp/2^l);
   temp = temp - a(k-l)*2^l;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

