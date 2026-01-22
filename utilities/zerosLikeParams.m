% zerosLikeParams.m
%
% Description:
%   Recursively create zero-valued placeholders that match the structure,
%   size, and data type of an input parameter container. Useful for
%   initializing optimizer moment estimates (e.g., Adam m/v) to mirror the
%   model parameter structure.
%
% Usage:
%   out = zerosLikeParams(p)
%
% Inputs:
%   p - parameter container (struct, cell, numeric array, dlarray, etc.)
%
% Outputs:
%   out - zeros with the same layout and 'like' type as p:
%         - If p is a struct: out is a struct with the same fields, each
%           field produced by a recursive call to zerosLikeParams.
%         - If p is a cell: out is a cell with each element produced by a
%           recursive call to zerosLikeParams.
%         - Otherwise: out = zeros(size(p), 'like', p).
%
% Notes:
%   - Preserves data class and GPU/CPU placement via the 'like' argument.
%   - Does not alter sparsity or complex-valued structure beyond what
%     zeros(...,'like',p) provides.
function out = zerosLikeParams(p)

if isstruct(p)
    fn = fieldnames(p);
    out = struct();
    for k = 1:numel(fn)
        f = fn{k};
        out.(f) = zerosLikeParams(p.(f));
    end
elseif iscell(p)
    out = cell(size(p));
    for k = 1:numel(p)
        out{k} = zerosLikeParams(p{k});
    end
else
    % numeric/dlarray/other: create zeros matching size/class/complexity/sparsity
    out = zeros(size(p), 'like', p);
end
end
