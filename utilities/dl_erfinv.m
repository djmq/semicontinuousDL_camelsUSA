% dl_erfinv.m
%
% dlarray/dlgradient-safe inverse error function using Newton iterations.
%
% Usage:
%   y = dl_erfinv(x)
%
% Input:
%   x - values in (-1,1) (dlarray or numeric)
%
% Output:
%   y - elementwise erfinv(x)
function y = dl_erfinv(x)

one  = cast(1, 'like', x);
tiny = cast(1e-12, 'like', x);

% clamp input to open interval (-1+eps, 1-eps)
x = min(max(x, -one + tiny), one - tiny);

% Winitzki initial approximation (good starting point)
a = cast(0.140012, 'like', x);   % Winitzki constant
ln1mx2 = log(1 - x.^2);
init = sign(x) .* sqrt( sqrt( (2/(pi*a) + ln1mx2/2).^2 - ln1mx2/a ) - ...
    (2/(pi*a) + ln1mx2/2) );

y = init;

% Newton iterations: solve f(y)=erf(y)-x = 0
% f'(y) = 2/sqrt(pi) * exp(-y^2)
sqrtpi = sqrt(pi);
for k = 1:4
    fy = erf(y) - x;
    fpy = (2 ./ sqrtpi) .* exp(- y .^ 2);
    y = y - fy ./ (fpy + eps(y));   % eps(y) to avoid div-by-zero
end

end
