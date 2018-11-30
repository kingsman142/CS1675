function y_pred_final = adaboost(X_train, y_train, X_test, iters)
    N = size(X_train, 1); % number of training samples
    M = size(X_test, 1); % number of testing samples
    
    w = ones(N, 1) .* (1.0/N); % initialize all weights of the training samples to 1/N
    preds = zeros(M, iters);
    alphas = zeros(iters, 1);
    y_pred_final = zeros(M, 1);
    
    for m = 1:iters % m rounds (different from M, which is the number of test samples)
        [correct_train, y_pred] = decision_stump_set(X_train, y_train, w, X_test); % find the best decision stump classifier
        preds(:, m) = y_pred; % the hypothesis for all test samples for the m_th iteration
        I = -(correct_train - 1); % incorrectly classified samples become 1; correctly classified samples become 0
        epsilon = dot(w, I) / sum(w);
        alpha = log((1.0 - epsilon) / (epsilon + 0.00000001)); % add the small value to epsilon so the denominator is never 0
        if alpha < 0 % don't allow negative alpha values
            alpha = 0;
        end
        alphas(m, :) = alpha; % keep the alpha value for this iteration of hypotheses
        
        w = w .* exp(alpha * I);
        w = w / norm(w, 1); % normalize weights
        y_pred_final = y_pred_final + (y_pred(m, :) .* alpha); % this is our y_M
    end
    
    for i = 1:M
        y_pred_final(i, :) = sign(dot(alphas, preds(i, :))); % just do this because removing it decreases our accuracy
    end
    y_pred_final = sign(y_pred_final); % now compute the actual sign so the predicted labels are -1 or 1
end