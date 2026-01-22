% get_fdc.m
%
% Compute the flow duration curve (sorted flows) and sorting indices.
%
% Usage:
%   [fdc, ix] = get_fdc(x)
%
% Inputs:
%   x  - vector of flows
%
% Outputs:
%   fdc - sorted flows in descending order
%   ix  - sorting indices such that fdc = x(ix)
function [out,ix] = get_fdc(x)

[out,ix] = sort(x,1,"descend");

end
