% fdc_fms.m
%
% Percent slope error of the middle section of the flow duration curve.
%
% Usage:
%   out = fdc_fms(obs, sim, lower, upper)
%
% Inputs:
%   obs, sim - vectors of observed and simulated flows
%   lower, upper - quantile fractions (e.g., 0.2, 0.7)
%
% Output:
%   out - percent slope difference
%
% Yilmaz, K. K., Gupta, H. V. & Wagener, T. A process-based diagnostic
% approach to model evaluation: Application to the NWS distributed
% hydrologic model. Water Resour. Res. 44, 1–18 (2008).  
%
% https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L351

function [out] = fdc_fms(obs,sim,lower,upper)

obs = get_fdc(obs);
sim = get_fdc(sim);

obs(obs==0) = 1e-6;
sim(sim<=0) = 1e-6;

qsm_lower = log(sim(floor(lower*size(sim,1))));
qsm_upper = log(sim(floor(upper*size(sim,1))));

qom_lower = log(obs(floor(lower*size(sim,1))));
qom_upper = log(obs(floor(upper*size(sim,1))));

out = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / ...
      (qom_lower - qom_upper + 1e-6);

out = 100 * out;

end