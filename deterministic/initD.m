%% Initialize model parameters function: Deterministic (D)
% initD  Initialize learnable parameters for deterministic LSTM model.
%
% Usage:
%   parameters = initD(numResponses, numFeatures, numHiddenUnits)
%
% Inputs:
%   numResponses    - dimensionality of network outputs
%   numFeatures     - number of input features
%   numHiddenUnits  - number of hidden units in LSTM
%
% Output:
%   parameters      - struct containing initialized weights and biases:
%                     .weights, .recurrentWeights, .bias,
%                     .fc.weights, .fc.bias
%
% Notes:
%   - Uses helper initializers: initializeGlorot_v3,
%     initializeOrthogonal_v3, initializeUnitForgetGate_v3,
%     initializeZeros_v3.
function [parameters] = initD(numResponses,numFeatures,numHiddenUnits)

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
