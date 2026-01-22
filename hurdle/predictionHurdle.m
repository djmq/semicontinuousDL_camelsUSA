%% Prediction function: Hurdle
% predictionHurdle  Run model forward pass for inference (no dropout).
%
% Usage:
%   [predictions1, predictions2] = predictionHurdle(modelFcn, parameters, XTest, dropoutRate)
%
% Inputs:
%   modelFcn     - handle to forward function (e.g., @forwardHurdle)
%   parameters   - parameter struct (from initHurdle / training)
%   XTest        - input sequence for inference (CTB dlarray/gpuArray-friendly)
%   dropoutRate  - dropout rate (not applied at inference; kept for signature)
%
% Outputs:
%   predictions1 - regression outputs (Y1)
%   predictions2 - classification logits (Y2)
%
% Notes:
%   - Calls modelFcn with doTraining = false.

function [predictions1,predictions2] = predictionHurdle(modelFcn,parameters,XTest,dropoutRate)

doTraining = false;
[predictions1,predictions2] = modelFcn(parameters,XTest,doTraining,dropoutRate);

end
