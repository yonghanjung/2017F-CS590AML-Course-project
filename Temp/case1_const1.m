function [ c,ceq ] = case1_const1( x, p_obs )
    c = -(p_obs*log(p_obs/x) + (1-p_obs)*log((1-p_obs)/(1-x)));
    ceq = [];
end
