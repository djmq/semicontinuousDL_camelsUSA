% camels_metrics.m
%
% Compute a collection of hydrologic model performance metrics.
%
% Usage:
%   out = camels_metrics(obs, sim)
%
% Inputs:
%   obs - observed flow vector (same length as sim)
%   sim - simulated flow vector (same length as obs)
%
% Output:
%   out - struct with fields:
%     nse      - NSE value or decomposition output from nse_decomposition
%     nnse     - normalized NSE (0..1) via bounded_efficiency
%     kgem     - KGE value or decomposition output from kge_decomposition
%     nkgem    - normalized KGE (-1..1) via bounded_efficiency
%     fhv      - percent bias in high flows (top 2%)
%     flv      - percent bias in low flows (lowest 30%)
%     fms      - percent slope error of middle FDC (0.2-0.7)
%     fhrmse   - RMSE in high flows (top 2%)
%     flrmse   - RMSE in low flows (lowest 30%)
%     fmrmse   - RMSE of middle FDC (0.2-0.7)
%     thrmse   - RMSE in high flows (top 2%) [synced]
%     tlrmse   - RMSE in low flows (lowest 30%) [synced]
%     tmrmse   - RMSE of middle FDC (0.2-0.7) [synced]
%     nrmse    - normalized RMSE (call to nrmse)
%
% Notes:
%   - This function assumes the helper metric functions are on the MATLAB path:
%       nse_decomposition, bounded_efficiency, kge_decomposition,
%       fdc_fhv, fdc_flv, fdc_fms, fdc_fhrmse, fdc_flrmse,
%       fdc_fmrmse, fdc_thrmse, fdc_tlrmse, fdc_tmrmse, nrmse
%   - Input validation is minimal; ensure obs and sim are same-length numeric vectors.

function out = camels_metrics(obs,sim)

% Basic input validation
if ~isvector(obs) || ~isvector(sim)
    error('obs and sim must be vectors.');
end
if numel(obs) ~= numel(sim)
    error('obs and sim must have equal length.');
end

% Ensure column vectors for consistency
obs = obs(:);
sim = sim(:);

% Compute NSE and normalized/bounded form
nse_val = nse_decomposition(obs, sim);
if isstruct(nse_val) && isfield(nse_val, 'NSE')
    nse_scalar = nse_val.NSE;
else
    nse_scalar = nse_val;
end
out.nse = nse_val;
out.nnse = bounded_efficiency(nse_scalar);

% Compute KGE and normalized/bounded form
kge_val = kge_decomposition(obs, sim);
if isstruct(kge_val) && isfield(kge_val, 'KGE')
    kge_scalar = kge_val.KGE;
elseif isnumeric(kge_val) && numel(kge_val) > 1
    kge_scalar = kge_val(end);
else
    kge_scalar = kge_val;
end
out.kgem = kge_val;
out.nkgem = bounded_efficiency(kge_scalar);

% FDC and RMSE style metrics (recommended defaults)
out.fhv     = fdc_fhv(obs, sim, 0.02);
out.flv     = fdc_flv(obs, sim, 0.3);
out.fms     = fdc_fms(obs, sim, 0.2, 0.7);
out.fhrmse  = fdc_fhrmse(obs, sim, 0.02);
out.flrmse  = fdc_flrmse(obs, sim, 0.3);
out.fmrmse  = fdc_fmrmse(obs, sim, 0.2, 0.7);

% Timing-synchronized FDC RMSEs
out.thrmse  = fdc_thrmse(obs, sim, 0.02);
out.tlrmse  = fdc_tlrmse(obs, sim, 0.3);
out.tmrmse  = fdc_tmrmse(obs, sim, 0.2, 0.7);

% Normalized RMSE
out.nrmse = nrmse(obs, sim);

end
