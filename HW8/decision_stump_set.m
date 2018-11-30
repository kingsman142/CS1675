function [correct_train, y_pred] = decision_stump_set(X_train, y_train, w_train, X_test)
    D = size(X_train, 2); % number of features in our dataset
    
    % dimension, threshold value, and over/under 'flip/side' of the training set with the lowest weighted error
    bestDimension = 0;
    bestThreshold = 0;
    bestOverUnder = 0;
    bestError = 0;
    
    for i = 1:D % for each feature
        thresholds = get_thresholds(X_train, i); % given the range of this feature's values, find 10 thresholds

        [bestThresholdNum, bestOverUnderNum, bestClassifierError] = get_best_threshold(X_train, y_train, w_train, i, thresholds');
        if bestClassifierError < bestError || bestError == 0 % we found a threshold on a feature with a new best error rate!
            bestError = bestClassifierError;
            bestOverUnder = bestOverUnderNum;
            bestThreshold = thresholds(bestThresholdNum, :);
            bestDimension = i;
        end
    end
    
    y_pred = predict_samples(X_test, bestDimension, bestThreshold, bestOverUnder); % predict the testing samples
    
    if bestOverUnder == 1 % over/under => positive/negative
        correct_train = (X_train(:, bestDimension) >= bestThreshold) == y_train; % 'positive' samples that are predicted as 'positive'
        correct_train = ((X_train(:, bestDimension) < bestThreshold) == -y_train) | correct_train; % 'negative' samples that are predicted as 'positive', then join this with the already created correct_train
    else % overUnder == 2 -- over/under => negative/positive
        correct_train = (X_train(:, bestDimension) >= bestThreshold) == -y_train; % 'negative' samples that are predicted as 'negative'
        correct_train = ((X_train(:, bestDimension) < bestThreshold) == y_train) | correct_train; % 'positive' samples that are predicted as 'positive', then join this with the already created correct_train
    end
    
end

% Predict the output labels for a given dataset
function y_pred = predict_samples(X_test, dimension, threshold, overUnder)
    if overUnder == 1 % over/under => positive/negative
        y_pred = X_test(:, dimension) >= threshold; % positive samples = 1; negative samples = 0
    else % overUnder == 2 -- over/under => negative/positive
        y_pred = X_test(:, dimension) < threshold; % positive samples = 1; negative samples = 0
    end
    y_pred = y_pred .* 2; % positive samples = 2; negative samples = 0
    y_pred = y_pred - 1; % positive samples = 1; negative samples = -1
end

% For a given dimension of a dataset and a set of thresholds to test, find
% the threshold with the lowest weighted error
function [bestThresholdNum, bestOverUnderNum, bestError] = get_best_threshold(X_train, y_train, weights, dimension, thresholds)
    numThresholds = size(thresholds, 2); % should be 10
    bestThresholdNum = 0;
    bestOverUnderNum = 0;
    bestError = 0;
    
    for i = 1:numThresholds % iterate over the thresholds
        threshold = thresholds(:, i);
        for j = 1:2
            if j == 1 % over/under => positive/negative
                positives = X_train(:, dimension) >= threshold; % samples predicted positive
                negatives = -(X_train(:, dimension) < threshold); % samples predicted negative
                posNegCombined = positives + negatives; % join the sets
                incorrect = double(posNegCombined ~= y_train); % find the samples predicted incorrectly
                error = dot(incorrect, weights); % incorrect samples * weight, then sum them together
                
                if error < bestError || bestError == 0
                    bestThresholdNum = i;
                    bestOverUnderNum = j;
                    bestError = error;
                end
            else % j == 2 -- over/under => negative/positive
                positives = X_train(:, dimension) < threshold; % samples predicted positive
                negatives = -(X_train(:, dimension) >= threshold); % samples predicted negative
                posNegCombined = positives + negatives; % join the sets
                incorrect = double(posNegCombined ~= y_train); % find the samples predicted incorrectly
                error = dot(incorrect, weights); % incorrect samples * weight, then sum them together
                
                if error < bestError || bestError == 0
                    bestThresholdNum = i;
                    bestOverUnderNum = j;
                    bestError = error;
                end
            end
        end
    end
end

% For a given dimension of a dataset, find the 10 equally spaced thresholds
function thresholds = get_thresholds(X_train, dimension)
    numThresholds = 10.0;
    minVal = min(X_train(:, dimension));
    maxVal = max(X_train(:, dimension));
    step = (maxVal - minVal) / (numThresholds + 1.0);
    
    thresholds = minVal:step:maxVal;
    thresholds = thresholds(2:11)';
end