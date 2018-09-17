function [w] = lr_solve_closed(X_train, y_train)
    w = pinv(X_train)*y_train;
end

