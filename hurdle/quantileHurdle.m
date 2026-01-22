%% Quantile function: Hurdle Burr-XII
% quantileHurdle  Compute quantiles for hurdle-based Burr XII in a dlarray-friendly way.
%
% Usage:
%   q = quantileHurdle(b, c, a, prob0, p)
%
% Inputs:
%   b, c, a    - Burr-XII parameters (scalars or arrays; dlarray/gpuArray-friendly)
%                Requirements: a > 0, b > 0, c > 0
%   prob0      - hurdle probability mass at zero (>0)
%   p          - target probabilities in [0,1] (same shape as b or broadcastable)
%
% Outputs:
%   q - quantiles (same shape as p), with:
%       - q = 0 for probabilities mapping into mass at zero
%       - finite positive quantile for interior probabilities
%       - large finite cap for probabilities very close to 1
%
% Notes:
%   - This implementation is elementwise and avoids host indexing so it is
%     compatible with dlarray and GPU arrays.

function q = quantileHurdle(b, c, a, prob0, p)
% quantileHurdle  dlarray-safe hurdle-based Burr XII quantile
% Inputs:
%  b, c, a    - Burr parameters (scalar or arrays). a>0, b>0, c>0
%  prob0      - hurdle probability for zero mass (>0)
%  p          - target probability in [0,1]
% Output:
%  q          - quantiles, same shape as b (dlarray/gpuArray friendly)

% Use p to set 'like' type for constants
likeArg = b;
tiny = cast(realmin(underlyingType(likeArg)), 'like', likeArg);
tiny = max(tiny, cast(1e-12, 'like', likeArg));
one  = cast(1, 'like', likeArg);

% Ensure capVal is same 'like' and broadcastable
capVal = cast(1e20, 'like', likeArg);

% validity mask for parameters
validParams = (a > 0) & (b > 0) & (c > 0) & (prob0 > 0);

% compute p_pbf = (p + prob0 - 1) ./ prob0
p_pbf = (p + prob0 - one) ./ prob0;

% clamp p_pbf for stable inversion, but keep masks for special cases
p_pbf_clamped = min(max(p_pbf, tiny), one - tiny);

% compute candidate quantile using inverse Burr (elementwise)
inner = (1 - p_pbf_clamped) .^ ( -1 ./ c ) - 1;
inner = max(inner, cast(0, 'like', likeArg));
q_candidate = a .* ( inner .^ (1 ./ b) );

% logical masks (elementwise)
is_nonpos = p_pbf <= 0;                      % quantile = 0
is_almost_one = p_pbf >= (one - tiny);       % quantile -> +Inf (use cap)
is_regular = ~(is_nonpos | is_almost_one);  % interior values

% default output
q = zeros(size(p), 'like', likeArg);

% apply parameter validity and blend results without host indexing
valid_nonpos  = validParams & is_nonpos;
valid_regular = validParams & is_regular;
valid_cap     = validParams & is_almost_one;

q = q + valid_nonpos  .* cast(0, 'like', likeArg);   % p_pbf <= 0 -> 0
q = q + valid_regular .* q_candidate;               % interior
q = q + valid_cap     .* capVal;                    % p_pbf ~ 1 -> finite cap

% Optionally mark invalid parameter locations as NaN instead of 0:
% invalidMask = ~validParams;
% q = q + invalidMask .* cast(nan, 'like', likeArg);

end
