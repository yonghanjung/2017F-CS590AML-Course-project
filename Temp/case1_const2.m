function [ c,ceq ] = case1_const2( x, p_obs, px )
    c = (p_obs*log(p_obs/x) + (1-p_obs)*log((1-p_obs)/(1-x))) - px;
    ceq = [];
end
