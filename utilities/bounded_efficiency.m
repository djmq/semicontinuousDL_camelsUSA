% bounded_efficiency.m
%
% Compute normalized and bounded versions of an efficiency metric.
%
% Usage:
%   [neff, beff] = bounded_efficiency(eff)
%
% Inputs:
%   eff - scalar or vector of efficiency values (e.g., NSE or KGE)
%
% Outputs:
%   neff - normalized efficiency in (0,1): neff = 1./(2 - eff)
%   beff - bounded efficiency in (-1,1): beff = eff ./ (2 - eff)
%
% Notes:
%   - Formulas from Nossent & Bauwens (2012) and Mathevet et al. (2006).
%   - Handles elementwise inputs. Avoid eff values exactly equal to 2 (division by zero).
%
% https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
%
%  Mathevet, Thibault; Michel, Claude; Andréassian, Vazken; Perrin, Charles
%  (2006). "A bounded version of the Nash–Sutcliffe criterion for better
%  model assessment on large sets of basins". IHS Publication 307: 211–220.  
%
% Nossent, J; Bauwens, W (2012). "Application of a normalized
% Nash–Sutcliffe efficiency to improve the accuracy of the
% Sobol'sensitivity analysis of a hydrological model". EGUGA: 237.

function [neff,beff] = bounded_efficiency(eff)

% ensure numeric input
arguments
    eff {mustBeNumeric}
end

% clamp values very close to 2 to avoid division-by-zero producing Inf
epsval = 1e-12;
eff = min(eff, 2 - epsval);

neff = 1 ./ (2 - eff);      % normalized (0,1) per Nossent & Bauwens (2012)
beff = eff ./ (2 - eff);    % bounded (-1,1) per Mathevet et al. (2006)

end