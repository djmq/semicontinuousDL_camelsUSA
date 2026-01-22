%% Initialize model parameters function: Hurdle
% initHurdle  Initialize learnable parameters for the hurdle LSTM model.
%
% Usage:
%   parameters = initHurdle(numResponsesRegression, numResponsesClassification, ...
%                           numFeatures, numHiddenUnits)
%
% Inputs:
%   numResponsesRegression     - outputs for regression head
%   numResponsesClassification - classes for classification head
%   numFeatures                - input feature dimension
%   numHiddenUnits             - number of LSTM hidden units
%
% Outputs:
%   parameters - struct containing initialized fields:
%                .weights, .recurrentWeights, .bias,
%                .fc1.weights, .fc1.bias, .fc2.weights, .fc2.bias
%
% Notes:
%   - Uses project initializers (initializeGlorot_v3, initializeOrthogonal_v3,
%     initializeUnitForgetGate_v3, initializeZeros_v3).

function [parameters] = initHurdle(numResponsesRegression,numResponsesClassification,numFeatures,numHiddenUnits)

% Create the initial hidden and cell states. Use the same initial hidden
% state and cell state for all observations.

% Create the learnable parameters for the LSTM operation
sz = [4*numHiddenUnits numFeatures]; 
numOut = 4*numHiddenUnits;
numIn = numFeatures;
parameters.weights = initializeGlorot_v3(sz,numOut,numIn);

sz = [4*numHiddenUnits numHiddenUnits]; 
parameters.recurrentWeights = initializeOrthogonal_v3(sz);

parameters.bias = initializeUnitForgetGate_v3(numHiddenUnits);

% Fully connected layer corresponding to the regression output.
sz = [numResponsesRegression numHiddenUnits];
numOut = numResponsesRegression;
numIn = numHiddenUnits;
parameters.fc1.weights = initializeGlorot_v3(sz,numOut,numIn);
parameters.fc1.bias = initializeZeros_v3([numResponsesRegression 1]);

% Fully connected layer corresponding to the classification output.
sz = [numResponsesClassification numHiddenUnits];
numOut = numResponsesClassification;
numIn = numHiddenUnits;
parameters.fc2.weights = initializeGlorot_v3(sz,numOut,numIn);
parameters.fc2.bias = initializeZeros_v3([numResponsesClassification 1]);

end
