% plot_points_w
% use: store your vector of true labels in a variable called Y, and your predicted labels in Y_pred
% legend: uses hollow circles for positives and filled for negatives, red for misclassified points and green for correctly classified ones

fprintf('Iteration %u, picked instance with coordinates: %u %u, w is now: %f %f\n', ct, X(sel, 1), X(sel, 2), w(1), w(2));

figure('Position', [300 300 600 600])
hold on

for i = 1:size(X, 1)
    if(Y(i) == 1)
        if(Y(i) == Y_pred(i))
            scatter(X(i, 1), X(i, 2), 'g');
        else
            scatter(X(i, 1), X(i, 2), 'r');
        end
    else
        if(Y(i) == Y_pred(i))
            scatter(X(i, 1), X(i, 2), 'filled', 'g');
        else
            scatter(X(i, 1), X(i, 2), 'filled', 'r');
        end
    end
end

% plot a grid over the points and vector
grid
axis([-5 5 -5 5])

% plot w as a vector
origin = [0 0];                             % origin
p = w * 5;                                  % coordinates of w, making it longer so it's more visible
quiver(origin(1), origin(2), p(1), p(2), 0) % w as a vector

% plot decision boundary
slope = -w(1)/w(2);                         % perpendicular to w
y_intercept = 0;                            % it passes through the origin
x = -5:1:5;
plot(x, slope*x+y_intercept, 'k');

% pausing
disp('pausing... (press any key to continue)');
pause
