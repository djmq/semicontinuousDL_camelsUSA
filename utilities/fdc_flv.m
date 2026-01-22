% fdc_flv.m
%
% Percent bias in low-flow volume (log-space based) for the lowest l fraction.
%
% Usage:
%   out = fdc_flv(obs, sim, l)
%
% Inputs:
%   obs - observed flow vector
%   sim - simulated flow vector (same length as obs)
%   l   - tail fraction (e.g., 0.3)
%
% Output:
%   out - percent bias in low-flow volume
%
% Yilmaz, K. K., Gupta, H. V. & Wagener, T. A process-based diagnostic
% approach to model evaluation: Application to the NWS distributed
% hydrologic model. Water Resour. Res. 44, 1–18 (2008).  
%
% https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L472

function [out] = fdc_flv(obs,sim,l)

obs = get_fdc(obs);
sim = get_fdc(sim);

n = numel(obs);
k = floor(l * n);
if k < 1
    out = 0;
    return;
end

obs_tail = obs(end-k+1:end);
sim_tail = sim(end-k+1:end);

% Avoid zeros / negative values
obs_tail(obs_tail == 0) = 1e-6;
sim_tail(sim_tail <= 0) = 1e-6;

obs_log = log(obs_tail);
sim_log = log(sim_tail);

qsl = sum(sim_log - min(sim_log));
qol = sum(obs_log - min(obs_log));

out = -1 * (qsl - qol) / (qol + 1e-6);
out = 100 * out;

end
