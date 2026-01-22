%% Deterministic loss function: catchment-averaged NSE variant (D)
% lossD  Compute mean elementwise deterministic loss for predictions.
%
% Usage:
%   loss = lossD(Y, T, sb)
%
% Inputs:
%   Y   - predictions
%   T   - targets
%   sb  - target standard deviation or scaling (TStd)
%
% Output:
%   loss - scalar mean loss (mean of ((Y-T).^2) ./ ((sb+0.1).^2))
%
% Notes:
%   - Based on catchment-averaged NSE formulation referenced in code
%     comments (Kraftert et al. and Acuña Espinoza et al.).
function loss = lossD(Y,T,sb)

% catchment-averaged NSE from Kraztert et al. (2019) based on Eq. 1 in
% Acuña Espinoza et al. (2024)

% https://doi.org/10.5194/hess-23-5089-2019
% https://doi.org/10.5194/hess-28-2705-2024

loss = mean(((Y-T).^2)./((sb+0.1).^2));

end
