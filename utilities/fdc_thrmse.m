% fdc_thrmse.m
%
% RMSE in high flows (head) with synchronized timing.
%
% Usage:
%   out = fdc_thrmse(obs, sim, h)
%
% Inputs:
%   obs - observed flow vector
%   sim - simulated flow vector (same length as obs)
%   h   - head fraction (e.g., 0.02)
%
% Output:
%   out - scalar RMSE for the high-flow head
%
% Song, Y. et al. When ancient numerical demons meet physics-informed
% machine learning: adjoint-based gradients for implicit differentiable
% modeling. Hydrol. Earth Syst. Sci. 28, 3051–3077 (2024).  
%
% https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L419

function [out] = fdc_thrmse(obs,sim,h)

[obs,ix] = get_fdc(obs);
sim = sim(ix);

n = numel(obs);
k = max(floor(h * n), 1);

obs_head = obs(1:k);
sim_head = sim(1:k);

out = sqrt(mean((sim_head - obs_head).^2));

end
