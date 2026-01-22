% fdc_tmrmse.m
%
% RMSE of the middle section of the flow duration curve with synchronized timing.
%
% Usage:
%   out = fdc_tmrmse(obs, sim, lower, upper)
%
% Inputs:
%   obs   - observed flow vector
%   sim   - simulated flow vector (same length as obs)
%   lower - lower quantile fraction (e.g., 0.2)
%   upper - upper quantile fraction (e.g., 0.7)
%
% Output:
%   out - scalar RMSE
%
% Yilmaz, K. K., Gupta, H. V. & Wagener, T. A process-based diagnostic
% approach to model evaluation: Application to the NWS distributed
% hydrologic model. Water Resour. Res. 44, 1–18 (2008).  
%
% Song, Y. et al. When ancient numerical demons meet physics-informed
% machine learning: adjoint-based gradients for implicit differentiable
% modeling. Hydrol. Earth Syst. Sci. 28, 3051–3077 (2024). 

% https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L351
function [out] = fdc_tmrmse(obs,sim,lower,upper)

[obs,ix] = get_fdc(obs);
sim = sim(ix);

qsm_lower = sim(floor(lower*size(sim,1)));
qsm_upper = sim(floor(upper*size(sim,1)));

qom_lower = obs(floor(lower*size(sim,1)));
qom_upper = obs(floor(upper*size(sim,1)));

out = sqrt(mean((((qsm_lower - qsm_upper) - (qom_lower - qom_upper))).^2));

end