% distributionalParametersRG.m
%
% Description:
%   Transform raw network outputs to distributional parameters for the
%   Rectified Gaussian (RG) observation model. Applies a safe softplus
%   to ensure positive scale (sigma) and leaves location (mu) unchanged.
%
% Usage:
%   [mu, sigma] = distributionalParametersRG(Y)
%
% Inputs:
%   Y   - untransformed network outputs, shape [2, N] where
%         row 1 -> raw mu, row 2 -> raw sigma
%
% Outputs:
%   mu    - transformed location parameter (same shape as Y(1,:))
%   sigma - transformed positive scale parameter (same shape as Y(2,:))
%
% Notes:
%   - Designed to work with untransformed outputs produced by predictionRG.
%   - Uses dlarray-compatible operations (dl_softplus) and preserves type/device.
function [mu, sigma] = distributionalParametersRG(Y)

% see: lossRG.m

i_mu = 1; % location parameter (mean)
i_sigma = 2; % scale parameter (standard deviation)

% Location (mu) does not need any transformation
mu = Y(i_mu,:);

% Scale (sigma) must be greater than 0
% add_eps = 1e-5; % adjust as needed
add_eps = cast(1e-5, 'like', Y);
sigma = add_eps + dl_softplus(Y(i_sigma,:)); % safe softplus


end
