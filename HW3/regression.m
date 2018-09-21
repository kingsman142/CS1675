wineData = dlmread("winequality-red.csv", ";");
[trainInd, testInd, valInd] = dividerand(size(wineData, 1), 0.5, 0.5, 0.0);

[trainX, trainY, testX, testY] = preprocessData(trainInd, testInd, wineData);

% Closed-form Solution
closed_w = lr_solve_closed(trainX, trainY);
closed_pred = lr_predict(testX, closed_w);
closed_score = score(closed_pred, testY);
fprintf("LR Closed Score: %.4f\n", closed_score);

% Gradient Descent Solution
gradient_w = lr_solve_gd(trainX, trainY, 50, 0.001); % 50 iterations and .001 learning rate
gradient_pred = lr_predict(testX, gradient_w);
gradient_score = score(gradient_pred, testY);
fprintf("LR Gradient Score: %.4f\n", gradient_score);

% Calculate L2 error rate
function err = score(pred, testY)
    err = norm(pred - testY);
end

function [trainX, trainY, testX, testY] = preprocessData(trainInd, testInd, data)
    trainN = size(trainInd, 2);
    testN = size(testInd, 2);
    D = size(data, 2);
    trainX = zeros(trainN, D-1);
    trainY = zeros(trainN, 1);
    testX = zeros(testN, D-1);
    testY = zeros(testN, 1);
    
    % split out the training and testing samples
    for i = 1:trainN
        trainX(i, :) = data(trainInd(i), 1:D-1);
        trainY(i) = data(trainInd(i), D);
    end
    for i = 1:testN
        testX(i, :) = data(testInd(i), 1:D-1);
        testY(i) = data(testInd(i), D);
    end
    [means, stds] = computeMeanStd(trainX);
    
    % normalize training and testing data
    [trainX] = normalizeData(trainX, means, stds);
    [testX] = normalizeData(testX, means, stds);
    
    % add a bias to the training and testing data
    [trainX] = addBiasFeature(trainX);
    [testX] = addBiasFeature(testX);
end

function [newData] = addBiasFeature(data)
    N = size(data, 1);
    newData = [data ones(N, 1)];
end

% Standardize every feature of the data based on its mean and std for each
% sample
function [newData] = normalizeData(data, means, stds)
    N = size(data, 1);
    D = size(data, 2);
    newData = zeros(N, D);
    for i = 1:D
        newCol = (data(:, i) - means(i)) / stds(i);
        newData(:, i) = newCol;
    end
end

% Calculate the mean and std of each feature, or column, of the data
function [means, stds] = computeMeanStd(trainData)
    D = size(trainData, 2);
    means = zeros(D, 1);
    stds = zeros(D, 1);
    for i = 1:D
        means(i) = mean(trainData(:, i));
        stds(i) = std(trainData(:, i));
    end
end