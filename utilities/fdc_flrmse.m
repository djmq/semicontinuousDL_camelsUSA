% fdc_flrmse.m
%
% RMSE in low flows (tail) using non-log values.
%
% Usage:
%   out = fdc_flrmse(obs, sim, l)
%
% Inputs:
%   obs - observed flow vector
%   sim - simulated flow vector (same length as obs)
%   l   - tail fraction (e.g., 0.3)
%
% Output:
%   out - scalar RMSE for the low-flow tail
%
% Song, Y. et al. When ancient numerical demons meet physics-informed
% machine learning: adjoint-based gradients for implicit differentiable
% modeling. Hydrol. Earth Syst. Sci. 28, 3051–3077 (2024).  
%
% https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/evaluation/metrics.py#L419

function [out] = fdc_flrmse(obs,sim,l)

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

out = sqrt(mean((sim_tail - obs_tail).^2));

end
