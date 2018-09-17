function [w] = lr_solve_gd(X_train, y_train, iters, eta)
    D = size(X_train, 2);
    w = rand(D, 1);
    for i = 1:iters
        pred = lr_predict(X_train, w);
        disp(size(pred));
        disp(size(y_train));
        grad = gradient(pred);
        disp(size(w));
        disp(size(eta));
        disp(size(grad));
        w = w - eta*grad;
    end
end