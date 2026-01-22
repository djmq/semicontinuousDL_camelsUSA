% dl_softplus.m
%
% Stable softplus for dlarray (elementwise).
%
% Usage:
%   s = dl_softplus(x)
%
% Input:
%   x - dlarray or numeric
%
% Output:
%   s - same size as x
function s = dl_softplus(x)

s = max(x, 0) + log(1 + exp(-abs(x)));

end
