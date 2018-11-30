data = [-1 -4 1; -3 -1 1; -3 -2 1; -2 1 1; -1 -1 1; 4 5 -1; 1 3 -1; 4 0 -1; 3 2 -1; 5 3 -1]; % our data points
zeta = 0.1; % learning rate
w = rand(2, 1); % weights
N = size(data, 1); % number of samples

X = data(:, 1:2); % sample feature vectors
Y = data(:, 3); % sample classes
ct = 0; % iteration count for the plotting functionality

for i = 1:N % iterate over all the samples
    Y_pred = predict(X, w); % predicted classes
    [acc, sel] = compute_accuracy(Y_pred, Y); % compute accuracy and random index of incorrectly classified sample
    plot_points_w(); % plot the line of maximum margin on a plot
    w = w - zeta*X(i, :)'; % update the weights using the learning parameter
    ct = ct + 1;
end

% Predict the classes of the samples in X given weights w
function pred = predict(X, w)
    N = size(X, 1);
    pred = zeros(N, 1);
    for i = 1:N
        pred(i) = sign(dot(X(i, :), w));
    end
end

% Compute accuracy using the predicted classes and the actual classes
function [acc, incorrect_index] = compute_accuracy(pred, actual)
    N = size(pred, 1);
    correct = 0;
    incorrect_indices = [];
    for i = 1:N
        if pred(i) == actual(i)
            correct = correct + 1;
        else
            incorrect_indices = [incorrect_indices; i]; % add to the list of incorrect indices
        end
    end
    
    indices = size(incorrect_indices, 1); % number of incorrect indices
    if indices > 0 % there is at least 1 incorrect index, so find that sample's index in the dataset!
        incorrect_index = incorrect_indices(floor((indices-1)*rand())+1);
    else % there are no incorrectly classified samples, but the plotter needs a positive index to work correctly
        incorrect_index = 1;
    end
    acc = correct / N; % compute the accuracy
end