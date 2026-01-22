% initRG.m
%
% Description:
%   Initialize model parameters for the RG model (LSTM weights, recurrent
%   weights, biases and a fully connected output layer). Uses Glorot and
%   orthogonal initializations and a forget-gate bias initializer.
%
% Usage:
%   parameters = initRG(numResponses, numFeatures, numHiddenUnits)
%
% Inputs:
%   numResponses    - number of response outputs (2 for RG: mu, sigma)
%   numFeatures     - number of input features to LSTM
%   numHiddenUnits  - number of hidden units in LSTM
%
% Output:
%   parameters - struct containing initialized fields:
%                weights, recurrentWeights, bias, fc.weights, fc.bias
function [parameters] = initRG(numResponses,numFeatures,numHiddenUnits)

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

% Fully connected layer corresponding to the output.
sz = [numResponses numHiddenUnits];
numOut = numResponses;
numIn = numHiddenUnits;
parameters.fc.weights = initializeGlorot_v3(sz,numOut,numIn);
parameters.fc.bias = initializeZeros_v3([numResponses 1]);


end
