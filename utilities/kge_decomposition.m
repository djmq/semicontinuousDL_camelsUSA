% kge_decomposition.m
%
% Description:
%   Compute the Kling-Gupta Efficiency (KGE) and its decomposition
%   components following Gupta et al. (2009) and Kling et al. (2012).
%   Returns classic KGE, modified KGE using variability ratio gamma (KGEm),
%   Pearson correlation r, and the component ratios alpha, beta, gamma.
%
% Usage:
%   [kge, kgem, r, alpha, beta, gamma] = kge_decomposition(obs, sim)
%
% Inputs:
%   obs - observed values (vector)
%   sim - simulated values (vector), same size/orientation as obs
%
% Outputs:
%   kge   - Kling-Gupta Efficiency (classic formulation)
%   kgem  - Modified KGE using gamma (variability ratio) instead of alpha
%   r     - Pearson correlation coefficient between obs and sim
%   alpha - variability ratio: std(sim,1) / std(obs,1)
%   beta  - bias ratio: mean(sim) / mean(obs)
%   gamma - coefficient of variation ratio: (std(sim)/mean(sim)) / (std(obs)/mean(obs))
%
% Notes:
%   - Uses population standard deviation (flag 1) to match referenced papers.
%   - Input vectors should not contain NaNs; handle or remove NaNs before
%     calling this function.
%
% Gupta, H. V., Kling, H., Yilmaz, K. K. & Martinez, G. F. Decomposition
% of the mean squared error and NSE performance criteria: Implications for
% improving hydrological modelling. J. Hydrol. 377, 80–91 (2009).  

% Kling, H., Fuchs, M. & Paulin, M. Runoff conditions in the upper
% Danube basin under an ensemble of climate change scenarios. J. Hydrol.
% 424–425, 264–277 (2012).  

function [kge,kgem,r,alpha,beta,gamma] = kge_decomposition(obs,sim)

r = corrcoef(obs,sim);
r = r(1,2); % r above is a correlation matrix
alpha = std(sim,1)/std(obs,1);
beta = mean(sim)/mean(obs);

kge = 1 - sqrt( (r-1)^2 + (alpha - 1)^2 + (beta - 1)^2 );

gamma = (std(sim,1)/mean(sim)) / (std(obs,1)/mean(obs));

kgem = 1 - sqrt( (r-1)^2 + (gamma - 1)^2 + (beta - 1)^2 );

end
