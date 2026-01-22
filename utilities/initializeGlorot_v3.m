% initializeGlorot_v3.m
%
% Description:
%   Initialize a weight matrix using the Glorot (Xavier) uniform scheme.
%   Draws samples from U(-bound, bound) where bound = sqrt(6 / (numIn + numOut)).
%   Returns a dlarray labeled with format "CU".
%
% Usage:
%   weights = initializeGlorot_v3(sz, numOut, numIn)
%   weights = initializeGlorot_v3(sz, numOut, numIn, className)
%
% Inputs:
%   sz        - size vector or dimensions accepted by rand (e.g., [m,n])
%   numOut    - number of output units (used to compute the Glorot bound)
%   numIn     - number of input units (used to compute the Glorot bound)
%   className - (optional) numeric class name for the returned array (default: 'single')
%
% Output:
%   weights - dlarray of size sz with values drawn from U(-bound, bound),
%             of class className, labeled "CU"
%
% Notes:
%   - Set the RNG for reproducibility prior to calling this function.
%   - To place the result on GPU, convert the numeric array with gpuArray
%     before wrapping with dlarray, or convert the dlarray after creation.
function weights = initializeGlorot_v3(sz,numOut,numIn,className)

arguments
    sz
    numOut
    numIn
    className = 'single'
end

Z = 2*rand(sz,className) - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights,"CU");

end
