% nse_decomposition.m
%
% Description:
%   Compute the Nash–Sutcliffe Efficiency (NSE) and its decomposition
%   components (Pearson correlation r, alpha, beta) following Gupta et al.
%   (2009). Useful for diagnosing differences between observed and
%   simulated time series in hydrological modelling.
%
% Usage:
%   [nse, r, alpha, beta] = nse_decomposition(obs, sim)
%
% Inputs:
%   obs - observed values (vector of length N)
%   sim - simulated values (vector of length N), same orientation as obs
%
% Outputs:
%   nse   - Nash–Sutcliffe Efficiency (scalar)
%   r     - Pearson correlation coefficient between obs and sim (scalar)
%   alpha - ratio of standard deviations: std(sim,1) / std(obs,1)
%   beta  - bias normalized by std(obs,1): (mean(sim)-mean(obs))/std(obs,1)
%
% Notes:
%   - Uses population standard deviation (flag 1) to match the reference
%     formulation in Gupta et al. (2009).
%   - Input vectors should not contain NaNs; remove or handle NaNs before
%     calling this function.
%
% Gupta, H. V., Kling, H., Yilmaz, K. K. & Martinez, G. F. Decomposition
% of the mean squared error and NSE performance criteria: Implications for
% improving hydrological modelling. J. Hydrol. 377, 80–91 (2009).  

function [nse,r,alpha,beta] = nse_decomposition(obs,sim)

r = corrcoef(obs,sim);
r = r(1,2); % r above is a correlation matrix
alpha = std(sim,1)/std(obs,1);
beta = (mean(sim)-mean(obs))/std(obs,1);

nse = 2*alpha*r - alpha^2 - beta^2;

end
