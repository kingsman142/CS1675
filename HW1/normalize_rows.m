function [B] = normalize_rows(A)
    B = A ./ repmat(sum(A,2), 1, size(A,2));
end