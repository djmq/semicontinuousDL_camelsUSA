% initializeUnitForgetGate_v3.m
%
% Description:
%   Create a bias vector for an LSTM with the forget-gate bias initialized
%   to one (Matlab Deep Learning toolbox convention). Returns a dlarray
%   labeled along the channel dimension.
%
% Usage:
%   bias = initializeUnitForgetGate_v3(numHiddenUnits)
%   bias = initializeUnitForgetGate_v3(numHiddenUnits, className)
%
% Inputs:
%   numHiddenUnits - number of hidden units in the LSTM layer
%   className      - (optional) numeric class for the returned array (default: 'single')
%
% Output:
%   bias - dlarray column vector of length 4*numHiddenUnits with the
%          forget-gate entries (indices numHiddenUnits+1 : 2*numHiddenUnits)
%          set to 1 and other entries zero, labeled "C"
%
% Notes:
%   - Uses population convention for gate ordering consistent with MATLAB's
%     LSTM implementation (input, forget, cell, output).
%   - To place the result on GPU, convert the numeric array (or dlarray)
%     with gpuArray before or after calling this function.
function bias = initializeUnitForgetGate_v3(numHiddenUnits,className)

arguments
    numHiddenUnits
    className = 'single'
end

bias = zeros(4*numHiddenUnits,1,className);

idx = numHiddenUnits+1:2*numHiddenUnits; % Matlab Deep Learning toolbox
bias(idx) = 1;

bias = dlarray(bias,"C");

end
