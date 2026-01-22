% lossRG.m
%
% Description:
%   Compute the negative log-likelihood (NLL) for the rectified Gaussian
%   model. Handles two branches:
%     - postive target (T > 0): Gaussian NLL using mu, sigma
%     - zero target (T == 0): -log Phi(-mu/sigma)
%
% Usage:
%   loss = lossRG(Y, T)
%
% Inputs:
%   Y - raw model outputs (untransformed) where Y(1,:) = mu, Y(2,:) = sigma_raw
%   T - target values (same shape as mu), dlarray-compatible
%
% Output:
%   loss - scalar dlarray (mean NLL over all elements)
function loss = lossRG(Y,T)

% see: https://max.pm/posts/rectnorm_dist/

i_mu = 1; % location parameter (mean)
i_sigma = 2; % scale parameter (standard deviation)

% ensure dlarray-compatible numeric constants on same type/device
one = cast(1, 'like', T);
half = cast(0.5, 'like', T);
log_sqrt_2pi = cast(0.5 * log(2*pi), 'like', T);


% find where T > 0 (different loss branches needed to handle T=0 and T>0)
obsMask = (T > 0);        % logical dlarray
obsMaskF = cast(obsMask, 'like', T);   % 1 where T>0, 0 otherwise (same type/device)

% Location (mu) does not need any transformation
mu = Y(i_mu,:);

% Scale (sigma) must be greater than 0
% add_eps = 1e-5; % adjust as needed
add_eps = cast(1e-5, 'like', Y);
sigma = add_eps + dl_softplus(Y(i_sigma,:)); % safe softplus

% ---- observed NLL (T > 0) ----
z = (T - mu) ./ sigma;
nll_positive = half .* (z.^2) + log(sigma) + log_sqrt_2pi;   % same shape

% ---- censored NLL (T == 0): -log Phi(-mu/sigma) ----
x = - mu ./ sigma;   % argument for Phi(x) where x = -mu/sigma

% numeric precision helpers
prob_floor = cast(realmin(underlyingType(x)), 'like', x); % smallest positive normal
tiny = cast(prob_floor, 'like', x);
log_floor = cast(log(prob_floor), 'like', x);

% basic stable Phi and log(Phi) (dlarray safe)
u = - x ./ cast(sqrt(2), 'like', x);
phi_basic = cast(0.5, 'like', x) .* (one - erf(u));   % Phi(x)
phi_basic = max(phi_basic, tiny);                     % floor to avoid 0
log_phi_basic = log(phi_basic);

% ---- asymptotic branch for extreme negatives ----
threshold = cast(-8, 'like', x);   % tune between -6 and -12
mask_ext = x < threshold;           % logical dlarray
maskf_ext = cast(mask_ext, 'like', x);

% compute elementwise asymptotic approximation (finite everywhere)
delta = cast(1e-12, 'like', x);                % avoid div-by-zero
x_sq = x.^2;
invx2 = 1 ./ (x_sq + delta);
invx4 = invx2.^2;
series = 1 - invx2 + 3 .* invx4;
series = max(series, cast(1e-12, 'like', series));  % avoid log(0)

x_neg_pos = max(-x, tiny);                     % -x but >= tiny to keep log finite
log_lead = -0.5 .* x_sq - log(x_neg_pos) - cast(0.5*log(2*pi), 'like', x);
log_asym_elem = log_lead + log(series);

% Build full-shape log_asym_full (elementwise; values are finite
% everywhere) 
log_asym_full = log_asym_elem;

% Blend safe: use asymptotic where mask_ext true, else basic
log_phi = maskf_ext .* log_asym_full + (one - maskf_ext) .* log_phi_basic;

% Final floor to guarantee finite values
log_phi = max(log_phi, log_floor);

nll_censored = - log_phi;

% ---- combine observed and censored contributions ----
nll_per_elem = obsMaskF .* nll_positive + (one - obsMaskF) .* nll_censored;

% reduce to scalar loss (mean)
loss = mean(nll_per_elem, 'all');


end
