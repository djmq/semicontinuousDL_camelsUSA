%% Distributional parameters from model predictions function: Hurdle
% distributionalParametersHurdle  Compute Burr-XII parameters and class probs.
%
% Usage:
%   [b, c, a, p0] = distributionalParametersHurdle(Y1, Y2, q)
%
% Inputs:
%   Y1        - regression head outputs (numParams x Batch, dlarray/gpuArray-friendly)
%   Y2        - classification logits (numClasses x Batch)
%   q         - hyperparameter controlling xi mapping to ensure bc > 1
%
% Outputs:
%   b         - Burr parameter b (positive)
%   c         - Burr parameter c (positive, derived to ensure bc>1)
%   a         - Burr parameter a (positive)
%   p0        - class probabilities (numClasses x Batch, softmax of Y2)
%
% Notes:
%   - Uses dl_softplus and sigmoid to produce positive parameters in a
%     manner compatible with automatic differentiation.
%   - Assumes parameter indices in Y1: i_b = 1, i_c = 2, i_a = 3.

function [b, c, a, p0] = distributionalParametersHurdle(Y1,Y2,q)

% This function works with the output from the (untransformed) output
% from predictionHurdle.m.
% see: lossHurdle.m

i_b = 1; % scale parameter 
i_c = 2; % shape parameter
i_a = 3; % shape parameter

% small stabilizers and device/type aware constants
add_eps = cast(1e-5, 'like', Y1);

% a, b must be greater than 0
b = add_eps + dl_softplus(Y1(i_b,:));
a = add_eps + dl_softplus(Y1(i_a,:));

% 0 < xi < 1 so that c is greater than 0 and bc > 1 (add safety factor for
% log-likelihood)
sig_eps = cast(0.1, 'like', Y1);
xi = (1/q - sig_eps) * sigmoid(Y1(i_c,:)) + sig_eps; % ensure 0<xi<1
c = 1./(xi.*b) + add_eps;

% Y2 is C x B numeric matrix
Y2m = Y2 - max(Y2, [], 1);            % subtract max per column for stability
expY = exp(Y2m);
p0 = expY ./ sum(expY, 1);        % C x B, columns sum to 1

end
