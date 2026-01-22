% initializeZeros_v3.m
%
% Description:
%   Create a zero-initialized array of a specified numeric class and wrap it
%   as a dlarray labeled along the channel dimension. Useful for
%   initializing learnable parameters or buffers for custom training loops.
%
% Usage:
%   parameter = initializeZeros_v3(sz)
%   parameter = initializeZeros_v3(sz, className)
%
% Inputs:
%   sz        - size vector or size arguments accepted by zeros (e.g., [h,w,c] or 1x3 vector)
%   className - (optional) numeric class name for the returned array (char or string).
%               Default: 'single'
%
% Output:
%   parameter - dlarray of zeros with the requested size and class, labeled "C"
%
% Notes:
%   - The function creates the numeric array with zeros(sz,className) and then
%     converts it to a dlarray. If you need a different labeling or no labels,
%     adjust or remove the dlarray call.
%   - If you require GPU storage, call gpuArray on the numeric array before
%     wrapping with dlarray, or use dlarray on a gpuArray.
function parameter = initializeZeros_v3(sz,className)

arguments
    sz
    className = 'single'
end

parameter = zeros(sz,className);
parameter = dlarray(parameter,"C");

end
