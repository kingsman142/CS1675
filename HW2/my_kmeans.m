function [ids, means, ssd] = runKMeans(A, K, iters)
    N = size(A, 1);
    D = size(A, 2);
    ids = zeros(N, 1); % Cluster ID of all N data points
    means = generateRandomCenters(A, K, D); % Array of K cluster centers with D dimensions
    for i = 1:iters
        ids = assignDataPoints(A, N, K, means);
        means = calcNewClusterCenters(A, K, D, means, ids);
        ssd = calcSSD(A, N, means, ids);
    end
end

% Randomly generate cluster centers
function means = generateRandomCenters(A, K, D)
    means = zeros(K, D);
    for i = 1:D
       minVal = min(A(:,i));
       maxVal = max(A(:,i));
       randVals = (maxVal - minVal).*rand(K, 1) + minVal; % generate random values in a range from min to max
       means(:,i) = randVals;
    end
end

% Assign each data point to its closest cluster
function ids = assignDataPoints(A, N, K, means)
    ids = zeros(N, 1);
    for i = 1:N
       dataPoint = A(i);
       closestCluster = 1;
       closestDistance = pdist2(dataPoint, means(1), 'euclidean');
       for cluster = 2:K
           distance = pdist2(dataPoint, means(cluster), 'euclidean');
           if distance < closestDistance
               closestCluster = cluster;
               closestDistance = distance;
           end
       end
       ids(i) = closestCluster;
    end
end

% Recompute the cluster centers
function means = calcNewClusterCenters(A, K, D, means, ids)
    for i = 1:K
       clusterIndices = find(ids == i);
       if size(clusterIndices,1) > 0
           newClusterMean = zeros(1, D);
           for j = 1:size(clusterIndices)
               dataPointIndex = clusterIndices(j);
               newClusterMean = newClusterMean + A(dataPointIndex, :);
           end
           newClusterMean = newClusterMean ./ size(clusterIndices,1);
           means(i,:) = newClusterMean;
       end
    end
end

% Calculate the sum of squared distances of the data points to their clusters
function ssd = calcSSD(A, N, means, ids)
    ssd = 0;
    for dataPointIndex = 1:N
        distance = pdist2(A(dataPointIndex), means(ids(dataPointIndex)), 'euclidean'); % compute euclidean distance
        ssd = ssd + distance.^2;
    end
end