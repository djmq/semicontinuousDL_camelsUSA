% initializeOrthogonal_v3.m
%
% Description:
%   Initialize a weight matrix with an orthogonal (or semi-orthogonal)
%   random basis using QR factorization of a Gaussian matrix. The sign
%   ambiguity of Q is resolved so that R has positive diagonal entries,
%   producing a deterministic orthogonal matrix up to the random seed.
%   The result is returned as a dlarray labeled with format "CU".
%
% Usage:
%   weights = initializeOrthogonal_v3(sz)
%   weights = initializeOrthogonal_v3(sz, className)
%
% Inputs:
%   sz        - size vector or two-element size argument accepted by randn.
%               For example, [m,n] produces an m-by-n matrix. If sz is
%               scalar, randn produces an sz-by-sz matrix.
%   className - (optional) numeric class name for the random matrix
%               (char or string). Default: 'single'
%
% Output:
%   weights - dlarray containing the orthogonal (or semi-orthogonal)
%             matrix of class className, labeled "CU"
%
% Notes:
%   - Uses randn to draw a Gaussian matrix and qr(...,0) for the economy-size
%     QR. For m > n, Q is m-by-n with orthonormal columns; for m <= n, Q is
%     square orthogonal.
%   - To ensure reproducibility, set the RNG prior to calling this function.
%   - To place the result on GPU, call gpuArray on Z before QR or convert
%     weights with gpuArray after creation.
function weights = initializeOrthogonal_v3(sz,className)

arguments
    sz
    className = 'single'
end

Z = randn(sz,className);
[Q,R] = qr(Z,0);

D = diag(R);
Q = Q * diag(D ./ abs(D));

weights = Q;
weights = dlarray(weights,"CU");

end
