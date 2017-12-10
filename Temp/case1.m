function [ c,ceq ] = case1( x, p_obs, px )
%CASE1 Summary of this function goes here
%   Detailed explanation goes here
[c1,ceq1] = case1_const1(x,p_obs);
[c2,ceq2] = case1_const2(x,p_obs, px);

c = [c1;c2];
ceq = [ceq1;ceq2];

end

