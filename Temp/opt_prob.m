A = []; 
Aeq = [];
b = [];
beq = [];

p_obs = 0.168;
Hx = 0.2033; 
fun = @(x) case1(x,p_obs,Hx);

target = 1; 

if (target == -1),
    x0 = 0.2;
    problem.objective = @(x) x; 
elseif (target == 1),
    x0 = 0.99;
    problem.objective = @(x) -x;
end

options = optimoptions('fmincon','Display','iter');
problem.solver = 'fmincon';
problem.options = options;
problem.x0 = x0; 
problem.nonlcon = fun;
problem.lb = 0; 
problem.ub = 1; 

x = fmincon(problem)

con = @(x) (p_obs*log(p_obs/x) + (1-p_obs)*log((1-p_obs)/(1-x)));