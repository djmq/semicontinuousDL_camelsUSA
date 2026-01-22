% thresholdL2Norm.m
%
% Description:
%   Clip (rescale) a gradient array so its L2 norm does not exceed a
%   specified threshold. If the L2 norm is greater than the threshold, the
%   gradients are scaled down uniformly to match the threshold; otherwise
%   they are returned unchanged.
%
% Usage:
%   gradients = thresholdL2Norm(gradients, gradientThreshold)
%
% Inputs:
%   gradients         - numeric array (or dlarray) of gradients to clip
%   gradientThreshold - positive scalar specifying the maximum allowed L2 norm
%
% Output:
%   gradients - clipped gradients with same size and type as input
%
% Notes:
%   - Preserves data type and device when using dlarray or GPU arrays via
%     elementwise operations.
%   - Assumes gradientThreshold > 0.
function gradients = thresholdL2Norm(gradients,gradientThreshold)

gradientNorm = sqrt(sum(gradients(:).^2));
if gradientNorm > gradientThreshold
    gradients = gradients * (gradientThreshold / gradientNorm);
end

end
