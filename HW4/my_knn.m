function y_pred = my_knn(X_train, y_train, X_test, K)
    M = size(X_test, 1);
    y_pred = zeros(M, 1);
    for i = 1:M
        distances = calculateDistances(X_train, X_test(i, :));
        [~, indices] = kSmallestIndices(distances, K);
        predictedClass = findMajorityClass(indices, y_train, K);
        y_pred(i) = predictedClass;
    end
end

% Given k indices, determine the majority class of those k classes
function class = findMajorityClass(indices, y_train, K)
    classes = zeros(K, 1);
    for i = 1:K
        index = indices(i);
        classes(i) = y_train(index, 1);
    end
    
    class = mode(classes);
end

% Find the k indices of the smallest distances of an array
function [vals, indices] = kSmallestIndices(distances, K)
    [vals, indices] = mink(distances, K);
end

% Distance from a test sample to all training samples
function distances = calculateDistances(X_train, X_test)
    N = size(X_train, 1);
    distances = zeros(N, 1);
    for i = 1:N
        distances(i, 1) = pdist2(X_train(i, :), X_test);
    end
end