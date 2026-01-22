% forwardRG.m
%
% Description:
%   Forward pass through the RG model: LSTM -> optional dropout -> fully
%   connected output layer. Returns the raw (untransformed) outputs used
%   by distributionalParametersRG and lossRG.
%
% Usage:
%   Y = forwardRG(parameters, X, doTraining, dropoutRate)
%
% Inputs:
%   parameters   - struct containing LSTM and fc parameters
%   X            - input sequence dlarray, format 'CTB' expected
%   doTraining   - logical flag for training behaviour (enables dropout)
%   dropoutRate  - scalar dropout probability in (0,1)
%
% Output:
%   Y - raw network outputs (untransformed), shape [numResponses, miniBatch]
function [Y] = forwardRG(parameters,X,doTraining,dropoutRate)

% LSTM layer
weights = parameters.weights;
recurrentWeights = parameters.recurrentWeights;
bias = parameters.bias;
sz = [size(recurrentWeights,2) 1];
H0 = initializeInitialHiddenOrCellState(sz);
C0 = initializeInitialHiddenOrCellState(sz);
[~,Y] = lstm(X,H0,C0,weights,recurrentWeights,bias); % keep only the last output from the hidden state (Y)

% Dropout layer

if doTraining

    dropoutScaleFactor = 1 - dropoutRate;
    dropoutMask = (rand(size(Y), 'like', Y) > dropoutRate ) /...
        dropoutScaleFactor;
    Y = Y.*dropoutMask;

end

% Fully connected layer
weights = parameters.fc.weights;
bias = parameters.fc.bias;
Y = fullyconnect(Y,weights,bias);

end
