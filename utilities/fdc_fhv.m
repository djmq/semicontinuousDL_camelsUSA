% fdc_fhv.m
%
% Percent bias in high flows (head) for the top h fraction.
%
% Usage:
%   out = fdc_fhv(obs, sim, h)
%
% Inputs:
%   obs - observed flow vector
%   sim - simulated flow vector (same length as obs)
%   h   - head fraction (e.g., 0.02)
%
% Output:
%   out - percent bias in high flows
%
% Yilmaz, K. K., Gupta, H. V. & Wagener, T. A process-based diagnostic
% approach to model evaluation: Application to the NWS distributed
% hydrologic model. Water Resour. Res. 44, 1–18 (2008).  
%
% https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L419

function [out] = fdc_fhv(obs,sim,h)

obs = get_fdc(obs);
sim = get_fdc(sim);

n = numel(obs);
k = max(floor(h * n), 1);

obs_head = obs(1:k);
sim_head = sim(1:k);

den = sum(obs_head);
if den == 0
    out = 0;
    return;
end

out = sum(sim_head - obs_head) / den;
out = 100 * out;

end
