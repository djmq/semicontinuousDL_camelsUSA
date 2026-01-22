% initializeInitialHiddenOrCellState.m
%
% Description:
%   Create a zero-initialized dlarray intended for initial hidden or cell
%   states of recurrent layers. The output is a dlarray labeled along the
%   batch and channel (or other) dimensions using the "CB" format.
%
% Usage:
%   parameter = initializeInitialHiddenOrCellState(sz)
%   parameter = initializeInitialHiddenOrCellState(sz, className)
%
% Inputs:
%   sz        - size vector or dimensions accepted by zeros (e.g., [numHiddenUnits, batchSize])
%   className - (optional) numeric class name for the returned array (char or string).
%               Default: 'single'
%
% Output:
%   parameter - dlarray of zeros with the requested size and class, labeled "CB"
%
% Notes:
%   - To place the result on GPU, convert the numeric array with gpuArray
%     before wrapping with dlarray, or convert the dlarray after creation.
%   - If you prefer no labels or a different labeling, adjust or remove the
%     dlarray label.
function parameter = initializeInitialHiddenOrCellState(sz,className)

arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter,"CB");

end
