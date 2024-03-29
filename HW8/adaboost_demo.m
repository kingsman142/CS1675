pimaData = csvread('pima.data');
pimaData = pimaData(randperm(size(pimaData, 1)), :);

for iters = [10, 20, 50, 500]
    [adaboost_cumAcc, adaboostExtra_cumAcc] = crossFoldValidation(pimaData, iters);
    fprintf('Adaboost Normal: Average accuracy across all folds for iters = %d is %.4f\n', iters, adaboost_cumAcc);
    fprintf('Adaboost Extra: Average accuracy across all folds for iters = %d is %.4f\n\n', iters, adaboostExtra_cumAcc);
end

function [adaboost_cumAcc, adaboostExtra_cumAcc] = crossFoldValidation(data, iters)
    adaboost_cumAcc = 0.0;
    adaboostExtra_cumAcc = 0.0;
    for i = 1:10 % 10 folds
        test = data(i*76-75 : i*76, :); % test set = 76 samples
        train = extractTrain(data, i);
        [testX, testY] = splitXy(test); % split into X and Y
        [trainX, trainY] = splitXy(train); % split into X and Y
        trainY = (trainY .* 2) - 1; % convert all training output labels, which are 0, to -1
        testY = (testY .* 2) - 1; % convert all testing output labels, which are 0, to -1
        
        [means, stds] = computeMeanStd(trainX); % calculate means and stds for future normalization
        trainX = normalizeData(trainX, means, stds);
        testX = normalizeData(testX, means, stds);
        
        pred = adaboost(trainX, trainY, testX, iters);
        acc = computeAccuracy(testY, pred); % compute the score, or accuracy
        adaboost_cumAcc = adaboost_cumAcc + acc;
        
        pred = adaboost_extra(trainX, trainY, testX, iters);
        acc = computeAccuracy(testY, pred);
        adaboostExtra_cumAcc = adaboostExtra_cumAcc + acc;
    end
    adaboost_cumAcc = adaboost_cumAcc / 10.0; % average accuracy cross all 10 folds
    adaboostExtra_cumAcc = adaboostExtra_cumAcc / 10.0;
end

% Compute accuracy as number of correctly predicted samples out of all test
% samples
function acc = computeAccuracy(testY, predY)
    correct = 0;
    N = size(testY, 1);
    for i = 1:N
        if testY(i) == predY(i)
            correct = correct + 1;
        end
    end
    
    acc = correct / N;
end

% Extract the training dataset where fold i is the testing set
function train = extractTrain(data, i)
    beginTestIndex = i*76-75;
    endTestIndex = i*76;
    if beginTestIndex == 1 % folds 1 - 9
        train = data(endTestIndex+1 : end-8, :);
    elseif beginTestIndex == 760-75 % fold 0
        train = data(1:beginTestIndex-1, :);
    else % all folds surrounding test set fold i
        train = data(1:beginTestIndex-1, :);
        train = [train; data(endTestIndex+1 : end-8, :)];
    end
end

% Extract first D-1 dimensions for X and Dth dimension for Y
function [X, y] = splitXy(data)
    D = size(data, 2);
    X = data(:, 1:D-1);
    y = data(:, D);
end

% Normalize data given means and standard deviations of training set
function newData = normalizeData(data, means, stds)
    N = size(data, 1);
    D = size(means, 1);
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