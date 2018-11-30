function [W1, W2, error_over_time] = backward(X, y, M, iters, eta)
    N = size(X, 1); % num samples
    D = size(X, 2); % num features
    W1 = rand(M, D); % weights between input and hidden layer
    W2 = rand(1, M); % weights between hidden and output layer
    error_over_time = zeros(iters, 1); % error at every iteration after picking a random sample and performing backpropagation
    
    for i = 1:iters
        index = randi([1, N]); % select a random sample to use for updating the weights
        [y_pred, Z] = forward(X(index, :), W1, W2); % forward sample to calculate error for backpropagation
        error = abs(y_pred(1,:) - y(1,:)); % error of this sample in the final output layer
        error_over_time(i, :) = error;
        
        [delta_k, delta_j] = calc_delta(y_pred, y(index, :), Z, W2); % deltas used for the weight update
        [W1, W2] = update_weights(X(index, :), W1, W2, Z, delta_k, delta_j, eta); % update weights
    end
end

% Calculate the deltas, or error values, for the output and hidden neurons for back propagation
function [delta_k, delta_j] = calc_delta(y_pred, y, Z, W2)
    M = size(W2, 2); % number of hidden neurons
    delta_k = y_pred - y; % error at output neuron
    delta_j = zeros(M, 1); % error at hidden neurons
    for j = 1:M
        delta_j(j, :) = (1 - Z(j, :).^2) * W2(:, j) * delta_k; % calculate the error of a given hidden neuron
    end
end

% Update the weights of the neural network
function [W1, W2] = update_weights(X, W1, W2, Z, delta_k, delta_j, eta)
    W2 = W2 - eta * delta_k * Z'; % update weights between hidden and output layers
    W1 = W1 - eta * delta_j .* X; % update weights between input and hidden layers
end