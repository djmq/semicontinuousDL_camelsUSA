% gradientsRG.m
%
% Description:
%   Compute the training loss and gradients for the RG model. Evaluates the
%   forward pass with dropout enabled and returns the scalar loss and the
%   gradients structure compatible with adamupdate and zerosLikeParams.
%
% Usage:
%   [loss, gradients] = gradientsRG(parameters, X, T, dropoutRate)
%
% Inputs:
%   parameters  - model parameters struct
%   X           - input batch dlarray
%   T           - target batch dlarray
%   dropoutRate - dropout probability
%
% Outputs:
%   loss       - scalar dlarray loss (mean negative log-likelihood)
%   gradients  - gradients struct matching parameters fields
function [loss,gradients] = gradientsRG(parameters,X,T,dropoutRate)

doTraining = true;
Y = forwardRG(parameters,X,doTraining,dropoutRate);

loss = lossRG(Y,T);
gradients = dlgradient(loss,parameters);

end
