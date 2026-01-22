%% Gradient function: Deterministic (D)
% gradientsD  Compute loss and parameter gradients for deterministic model.
%
% Usage:
%   [loss, gradients] = gradientsD(parameters, X, T, TStd, dropoutRate)
%
% Inputs:
%   parameters   - struct of learnable parameters
%   X            - input sequence
%   T            - target values
%   TStd         - target standard deviation (used in lossD)
%   dropoutRate  - dropout probability used during training
%
% Outputs:
%   loss         - scalar dlarray loss (mean per-element loss)
%   gradients    - structure of gradients with same fields as parameters
%
% Notes:
%   - Calls forwardD with doTraining = true to apply dropout during
%     gradient computation. Uses dlgradient to obtain parameter gradients.
function [loss,gradients] = gradientsD(parameters,X,T,TStd,dropoutRate)

doTraining = true;
Y = forwardD(parameters,X,doTraining,dropoutRate);

loss = lossD(Y,T,TStd);
gradients = dlgradient(loss,parameters);

end
