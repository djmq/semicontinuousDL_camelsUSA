% predictionRG.m
%
% Description:
%   Wrapper to produce model predictions for test/validation data using the
%   forward model in evaluation mode (no dropout). Returns the raw
%   untransformed outputs which can be converted to distributional
%   parameters with distributionalParametersRG.
%
% Usage:
%   predictions = predictionRG(modelFcn, parameters, XTest, dropoutRate)
%
% Inputs:
%   modelFcn    - function handle (accelerated) to forwardRG or similar
%   parameters  - model parameters struct
%   XTest       - test input dlarray
%   dropoutRate - dropout value (unused when doTraining=false but passed)
%
% Output:
%   predictions - raw model outputs (untransformed)
function [predictions] = predictionRG(modelFcn,parameters,XTest,dropoutRate)

doTraining = false;
predictions = modelFcn(parameters, XTest, doTraining, dropoutRate);



end
