% fdc_fmrmse.m
%
% RMSE of the middle section of the flow duration curve.
%
% Usage:
%   out = fdc_fmrmse(obs, sim, lower, upper)
%
% Inputs:
%   obs, sim - observed and simulated flow vectors
%   lower    - lower quantile fraction (e.g., 0.2)
%   upper    - upper quantile fraction (e.g., 0.7)
%
% Output:
%   out - scalar RMSE for the middle section
%
% Yilmaz, K. K., Gupta, H. V. & Wagener, T. A process-based diagnostic
% approach to model evaluation: Application to the NWS distributed
% hydrologic model. Water Resour. Res. 44, 1–18 (2008).  
%
% Song, Y. et al. When ancient numerical demons meet physics-informed
% machine learning: adjoint-based gradients for implicit differentiable
% modeling. Hydrol. Earth Syst. Sci. 28, 3051–3077 (2024). 
%
% https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L351%

function [out] = fdc_fmrmse(obs,sim,lower,upper)

obs = get_fdc(obs);
sim = get_fdc(sim);

n = numel(obs);
i_lower = max(floor(lower * n), 1);
i_upper = min(max(floor(upper * n), 1), n);

qsm_lower = sim(i_lower);
qsm_upper = sim(i_upper);

qom_lower = obs(i_lower);
qom_upper = obs(i_upper);

out = sqrt(mean((((qsm_lower - qsm_upper) - (qom_lower - qom_upper))).^2));

end
