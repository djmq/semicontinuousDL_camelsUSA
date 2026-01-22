% modelGradientsCheck.m
%
% Description:
%   Inspect a gradients parameter struct and return a flag indicating if
%   any NaN values are present. Recurses one level into struct-valued
%   fields that themselves contain fields of numeric arrays. Intended for
%   quick sanity checks during training to detect invalid gradient values.
%
% Usage:
%   flag = modelGradientsCheck(gradients)
%
% Inputs:
%   gradients - struct of gradient arrays or nested structs where leaf
%               values are numeric arrays or dlarray objects
%
% Outputs:
%   flag - logical scalar, true if any NaN is found anywhere inspected,
%          false otherwise
%
% Notes:
%   - This function checks for NaN in numeric arrays. It treats non-struct
%     fields as arrays and checks them directly. If a field is a struct,
%     it checks that struct's fields (one level deep).
%   - Does not traverse cell arrays or deeper nested structs.
function [flag] = modelGradientsCheck(gradients)

flag = false;

gradientsContent = fieldnames(gradients);
for i = 1:numel(gradientsContent)

    g = gradients.(gradientsContent{i})(:);

    if isstruct(g)

        gradientsContentSub = fieldnames(g);

        for ii = 1:numel(gradientsContentSub)

            gg = g.(gradientsContentSub{ii})(:);

            if any(isnan(gg))
                flag = true;
                break

            end

        end

    else

        if any(isnan(g)) || flag
            flag = true;
            break

        end

    end


end

end
