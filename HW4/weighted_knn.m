function y_test = weighted_knn(X_train, y_train, X_test, sigma)
    M = size(X_test, 1);
    y_test = zeros(M, 1);
    uniqueClasses = unique(y_train);
    for i = 1:M
        distances = calculateDistances(X_train, X_test(i, :));
        weights = convertDistancesToWeights(distances, sigma);
        predictedClass = findMajorityWeightedClass(weights, y_train, uniqueClasses);
        y_test(i) = predictedClass;
    end
end

% Given weights of all data points to the test sample, calculate the
% majority weight class
function class = findMajorityWeightedClass(weights, y_train, uniqueClasses)
    numClasses = size(uniqueClasses, 1);
    classWeightsBins = zeros(numClasses, 1);
    N = size(y_train, 1);
    for i = 1:N
        sampleClass = find(uniqueClasses == y_train(i)); % index of uniqueClasses that matches this training sample's class
        classWeightsBins(sampleClass) = classWeightsBins(sampleClass) + weights(i);
    end
    
    [~, maxWeightedClassIndex] = max(classWeightsBins);
    class = uniqueClasses(maxWeightedClassIndex);
end

% Use formula we learned in class to convert distances to weights given a
% sigma
function weights = convertDistancesToWeights(distances, sigma)
    N = size(distances, 1);
    weights = zeros(N, 1);
    for i = 1:N
        weights(i) = exp(-distances(i)^2 / sigma^2);
    end
end

% Distance from a test sample to all training samples
function distances = calculateDistances(X_train, X_test)
    N = size(X_train, 1);
    distances = zeros(N, 1);
    for i = 1:N
        distances(i, 1) = pdist2(X_train(i, :), X_test);
    end
end