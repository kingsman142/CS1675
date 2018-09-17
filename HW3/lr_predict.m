function [y_pred] = lr_predict(X_test, w)
    M = size(X_test, 1);
    y_pred = zeros(M, 1);
    for i = 1:M
        y_pred(i) = dot(w, X_test(i, :));
    end
end