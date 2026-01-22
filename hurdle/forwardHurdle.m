%% Forward model function: Hurdle
% forwardHurdle  Run LSTM, optional dropout, and two head projections.
%
% Usage:
%   [Y1, Y2] = forwardHurdle(parameters, X, doTraining, dropoutRate)
%
% Inputs:
%   parameters   - struct of learnable parameters (LSTM and fc heads)
%   X            - input sequence (CTB dlarray/gpuArray-friendly)
%   doTraining   - logical flag; true applies dropout
%   dropoutRate  - dropout probability used during training
%
% Outputs:
%   Y1           - regression head outputs (numResponsesRegression x Batch)
%   Y2           - classification logits (numResponsesClassification x Batch)
%
% Notes:
%   - Uses initializeInitialHiddenOrCellState, lstm, fullyconnect.
%   - When doTraining is true, inverted Bernoulli dropout mask is applied.

function [Y1,Y2] = forwardHurdle(parameters,X,doTraining,dropoutRate)

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
    dropoutMask = (rand(size(Y), 'like', Y) > dropoutRate ) / dropoutScaleFactor;
    Y = Y.*dropoutMask;
end

% Fully connected layer for regression head
weights1 = parameters.fc1.weights;
bias1 = parameters.fc1.bias;
Y1 = fullyconnect(Y,weights1,bias1);

% Fully connected layer for classification head
weights2 = parameters.fc2.weights;
bias2 = parameters.fc2.bias;
Y2 = fullyconnect(Y,weights2,bias2); % logits

end
