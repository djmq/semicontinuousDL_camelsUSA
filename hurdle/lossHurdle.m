%% Loss function: Hurdle
% lossHurdle  Negative log-likelihood for hurdle-based Burr XII model.
%
% Usage:
%   loss = lossHurdle(Y1, Y2, T, q)
%
% Inputs:
%   Y1 - regression head outputs (numParams x Batch)
%   Y2 - classification logits (2 x Batch)
%   T  - target values (1 x Batch)
%   q  - hyperparameter used for xi mapping (scalar)
%
% Outputs:
%   loss - scalar dlarray representing negative log-likelihood (averaged)
%
% Notes:
%   - Parameterization ensures a, b, c > 0 via softplus/sigmoid and stabilizers.
%   - Uses safe clamping for probabilities and log-likelihood values to maintain
%     numerical stability and differentiability.
%   - The final loss is the mean CE for binary part plus mean normalized Burr XII NLL.

function loss = lossHurdle(Y1,Y2,T,q)

% loss = lossHurdle(Y1,Y2,T,q) returns the negative logarithm of the
% likelihood (NLL) between the target T and the estimated conditional
% distribution (components of Y) as the loss considering the hurdle-based
% Burr XII distribution.

i_b = 1; % scale parameter 
i_c = 2; % shape parameter (see Koutsoyiannis, 2023)
i_a = 3; % shape parameter

% Koutsoyiannis (2023) Stochastics of Hydroclimatic Extremes: A Cool Look
% at Risk (3rd ed.)

% small stabilizers and device/type aware constants               
prob_floor = cast(realmin(underlyingType(Y1)), 'like', Y1); % smallest positive normal
tiny = cast(prob_floor, 'like', T);        % safe floor for T and probabilities
add_eps = cast(1e-5, 'like', Y1);

% a, b must be greater than 0
b = add_eps + dl_softplus(Y1(i_b,:));
a = add_eps + dl_softplus(Y1(i_a,:));

% 0 < xi < 1 so that c is greater than 0 and bc > 1 (add safety factor for
% log-likelihood)
sig_eps = cast(0.1, 'like', Y1);
xi = (1/q - sig_eps) * sigmoid(Y1(i_c,:)) + sig_eps; % ensure 0<xi<1
c = 1./(xi.*b) + add_eps;

% compute log-likelihood for each observation

% see: 
% Okasha, M.K.; Matter, M.Y. On the three-parameter Burr type XII
% distribution and its application to heavy tailed lifetime data. J. Adv.
% Math. 2015, 10, 3249–3442 

T_safe = max(T, tiny);         % elementwise, preserves grads (subgradient)
term1 = (T_safe ./ a) .^ b;    % finite
log_like_safe = -(c + 1) .* log(1 + term1) + (b - 1) .* log(T_safe) ...
                - b .* log(a) + log(b) + log(c);

% clamp to moderate range (still differentiable via min/max subgrad)
maxval = cast(1e3, 'like', log_like_safe);
minval = cast(-1e3, 'like', log_like_safe);
log_like_safe = min(max(log_like_safe, minval), maxval);

% ----- Hurdle / binary part -----
t_zero = (T == 0);
t_one  = (T ~= 0);
T_onehot = [t_zero; t_one];     % 2 x B

% Y2: 2 x B logits
Y2m = Y2 - max(Y2,[],1);
logsumexp = log(sum(exp(Y2m),1)) + max(Y2,[],1);
logp0 = Y2 - logsumexp;

% binary cross-entropy
ce = - sum(T_onehot .* logp0, 1);   % 1 x B, format 'CB'

% ----- Combine: use mask multiplication  -----
t_one_f = cast(t_one, 'like', log_like_safe);
nll_cont = - log_like_safe;         % per-element contribution if T>0
nll_cont = t_one_f .* nll_cont;     % zero where T==0

nCont = sum(t_one_f,'all');
sumCont = sum(nll_cont,'all');
loss = sumCont./max(nCont, cast(1,'like',sumCont)) + mean(ce, 'all');

end





 