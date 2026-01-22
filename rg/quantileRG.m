function q = quantileRG(mu, sigma, p)
% quantileRG  dlarray-safe quantile for rectified Gaussian (RG)
% Inputs:
%  mu    - location (same shape-broadcastable)
%  sigma - positive scale (same shape-broadcastable)
%  p     - target probabilities in [0,1], scalar or array (dlarray/gpuArray OK)
% Output:
%  q     - quantile values, same shape as mu (0 where p <= Phi(-mu/sigma))

% Set type/device 'like' mu
likeArg = mu;
one  = cast(1, 'like', likeArg);
tiny = cast(realmin(underlyingType(likeArg)), 'like', likeArg);
tiny = max(tiny, cast(1e-12, 'like', likeArg)); % avoid underflow
sqrt2 = cast(sqrt(2), 'like', likeArg);

% compute x = -mu ./ sigma  (same as loss branch)
x = - mu ./ sigma;

% stable computation of Phi(x) = 0.5*(1 + erf(x / sqrt(2)))
u = x ./ sqrt2;
Phi_x = cast(0.5, 'like', likeArg) .* (one + erf(u));
Phi_x = min(max(Phi_x, tiny), one - tiny);  % keep in (tiny, 1-tiny)

% mask for censored region: quantile = 0 when p <= Phi_x
censoredMask = p <= Phi_x;
censoredMaskF = cast(censoredMask, 'like', likeArg);

% For inverse, clamp argument to erfinv in (-1+eps, 1-eps)
arg = min(max(2 .* p - one, -one + tiny), one - tiny); % 2p-1 clamped

% gaussian inverse: mu + sigma * sqrt(2) * erfinv(2p-1)
q_candidate = mu + sigma .* sqrt2 .* dl_erfinv(arg);

% blend results elementwise (no host indexing)
q = censoredMaskF .* cast(0, 'like', likeArg) + ...
(one - censoredMaskF) .* q_candidate;

end
