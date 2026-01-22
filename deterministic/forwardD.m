%% Forward model function: Deterministic (D)
% forwardD  Forward (prediction) pass of deterministic LSTM-based model.
%
% Usage:
%   Y = forwardD(parameters, X, doTraining, dropoutRate)
%
% Inputs:
%   parameters   - struct of learnable parameters:
%                  .weights, .recurrentWeights, .bias,
%                  .fc.weights, .fc.bias
%   X            - input sequence (observations in columns, time along rows)
%   doTraining   - logical scalar, true to apply dropout mask
%   dropoutRate  - dropout probability used during training
%
% Output:
%   Y            - network output after LSTM, optional dropout, fully
%                  connected layer (last time-step output)
%
% Notes:
%   - Uses initializeInitialHiddenOrCellState for H0 and C0.
%   - Keeps only the last LSTM output; uses elementwise dropout when
%     doTraining is true.
function [Y] = forwardD(parameters,X,doTraining,dropoutRate)

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
