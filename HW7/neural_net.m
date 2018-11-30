wineData = dlmread("winequality-red.csv", ";");
[trainInd, testInd, valInd] = dividerand(size(wineData, 1), 0.5, 0.5, 0.0);

[trainX, trainY, testX, testY] = preprocessData(trainInd, testInd, wineData);

iters = 1000000; % number of samples to train backpropagation on
eta = 0.0001; % learning rate
hiddenUnits = 30; % number of neurons in the only hidden layer
numTestSamples = size(testX, 1);

[W1, W2, error_over_time] = backward(trainX, trainY, hiddenUnits, iters, eta); % train the neural network on the train data

y_pred = zeros(numTestSamples, 1);
for i = 1:numTestSamples % iterate over all test samples
    [pred_class, Z] = forward(testX(i, :), W1, W2); % predict the output value of this sample
    y_pred(i, :) = pred_class;
end

error = score(y_pred, testY); % calculate the final score over the predicted test classes

fprintf('Error over time: \n');
disp(error_over_time);
fprintf('Total error on predictions: %.4f\n', error);

% Calculate Root Mean Squared Error (RMSE)
function err = score(pred, testY)
    err = sqrt(mean((pred - testY).^2));
end

% Preprocess the data
function [trainX, trainY, testX, testY] = preprocessData(trainInd, testInd, data)
    trainN = size(trainInd, 2); % number of train samples
    testN = size(testInd, 2); % number of test samples
    D = size(data, 2); % number of dimensions
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

% Append a 1 to the end of all feature vectors as a 'bias'
function [newData] = addBiasFeature(data)
    N = size(data, 1);
    newData = [data ones(N, 1)];
end

% Standardize every feature of the data based on its mean and std for each sample
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