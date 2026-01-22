% nrmse.m
%
% Description:
%   Compute normalized root mean square error (NRMSE) with two common
%   normalizations: by the population standard deviation and by the range.
%   Returns both normalized metrics for comparing observed and simulated
%   series.
%
% Usage:
%   [out1, out2] = nrmse(obs, sim)
%
% Inputs:
%   obs - observed values (vector)
%   sim - simulated or predicted values (vector), same size as obs
%
% Outputs:
%   out1 - RMSE normalized by population standard deviation: rmse / std(obs,1)
%   out2 - RMSE normalized by range: rmse / (max(obs) - min(obs))
%
% Notes:
%   - Uses population standard deviation (flag 1) to match common
%     normalization conventions.
%   - Input vectors should not contain NaNs; handle or remove NaNs before
%     calling this function.
function [out1,out2] = nrmse(obs,sim)

% normalized root mean square error

rmse = sqrt(mean((sim-obs).^2));

out1 = rmse/std(obs,1);
out2 = rmse/(max(obs)-min(obs));

end
