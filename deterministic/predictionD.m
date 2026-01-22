%% Model prediction function: Deterministic (D)
% predictionD  Run deterministic model in inference mode.
%
% Usage:
%   predictions = predictionD(modelFcn, parameters, XTest, dropoutRate)
%
% Inputs:
%   modelFcn     - function handle to forward model (e.g., @forwardD)
%   parameters   - struct of learned parameters
%   XTest        - test input sequence
%   dropoutRate  - dropout probability (ignored when doTraining=false)
%
% Output:
%   predictions  - model outputs (no dropout applied)
%
% Notes:
%   - Calls modelFcn with doTraining = false to disable dropout.
function [predictions] = predictionD(modelFcn,parameters,XTest,dropoutRate)

doTraining = false;
predictions = modelFcn(parameters, XTest, doTraining, dropoutRate);

end
