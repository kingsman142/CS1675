function [w] = lr_solve_gd(X_train, y_train, iters, eta)
    D = size(X_train, 2);
    w = rand(D, 1);
    for i = 1:iters
        pred = lr_predict(X_train, w); % predict the samples with current weights
        grad = calc_gradient(X_train, pred, y_train); % find the gradient of the weights
        w = w - eta*grad; % recalculate the new weights
    end
end

function grad = calc_gradient(X_train, pred, output)
    N = size(output, 1);
    D = size(X_train, 2);
    grad = zeros(D, 1);
    for i = 1:N
        grad(:, 1) = grad + ((pred(i, :) - output(i, :))*X_train(i, :)).'; % derivative of the least squares loss function
    end
    grad = grad / N; % average over all the samples
end