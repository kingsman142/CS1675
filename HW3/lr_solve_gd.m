function [w] = lr_solve_gd(X_train, y_train, iters, eta)
    D = size(X_train, 2);
    w = rand(D, 1);
    for i = 1:iters
        pred = lr_predict(X_train, w);
        grad = calc_gradient(X_train, pred, y_train);
        w = w - eta*grad;
    end
end

function grad = calc_gradient(X_train, pred, output)
    N = size(output, 1);
    D = size(X_train, 2);
    grad = zeros(D, 1);
    for i = 1:N
        grad(:, 1) = grad + ((pred(i, :) - output(i, :))*X_train(i, :)).';
    end
    grad = grad / N;
end