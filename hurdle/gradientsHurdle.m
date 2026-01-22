%% Gradient function: Hurdle
% gradientsHurdle  Compute loss and parameter gradients for hurdle model.
%
% Usage:
%   [loss, gradients] = gradientsHurdle(parameters, X, T, dropoutRate, q)
%
% Inputs:
%   parameters   - struct of learnable parameters
%   X            - input sequence
%   T            - target values
%   dropoutRate  - dropout probability used during training
%   q            - hyperparameter used in lossHurdle / parameter mapping
%
% Outputs:
%   loss         - scalar dlarray loss (mean per-element loss)
%   gradients    - structure of gradients with same fields as parameters
%
% Notes:
%   - Calls forwardHurdle with doTraining = true to apply dropout during
%     gradient computation. Uses dlgradient to obtain parameter gradients.

function [loss,gradients] = gradientsHurdle(parameters,X,T,dropoutRate,q)

doTraining = true;
[Y1,Y2] = forwardHurdle(parameters,X,doTraining,dropoutRate);
loss = lossHurdle(Y1,Y2,T,q);
gradients = dlgradient(loss,parameters);

end
