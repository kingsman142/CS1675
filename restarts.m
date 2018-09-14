function [ids, means, ssd] = runKMeansRestarts(A, K, iters, R)
    N = size(A, 1);
    D = size(A, 2);
    ids = zeros(N, 1);
    means = zeros(K, 1);
    ssd = -1;
    for i = 1:R % R restarts
        [newIds, newMeans, newSSD] = my_kmeans(A, K, iters);
        if newSSD < ssd || ssd == -1 % find the k-means simulation with the lowest SSD
            ids = newIds;
            means = newMeans;
            ssd = newSSD;
        end
        fprintf('Finished restart %d with SSD = %f\n', i, newSSD);
    end
    fprintf('\nMinimum SSD = %f\n\n', ssd);
end