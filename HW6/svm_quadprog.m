function [y_pred] = svm_quadprog(X_train, Y_train, X_test, C)
    D = size(X_train, 2); % dimensionality of our dataset
    M = size(X_test, 1); % number of testing samples
    N = size(Y_train, 1); % number of training samples
    
    % Set the parameters (H, f, A, b, Aeq, beq, lb, ub) for the quadratic
    % optimization problem
    H = zeros(N, N);
    for i = 1:N
        for j = 1:N
            H(i,j) = Y_train(i,:)*Y_train(j,:)*X_train(i,:)*X_train(j,:)'; % H = y_i*y^t_i * x_i*x^t_i
        end
    end
    f = -ones(N,1);
    
    % A * alpha <= b
    A = [];
    b = [];
    
    % Aeq * alpha = 0 => Aeq * alpha = beq
    Aeq = Y_train';
    beq = 0;
    lb = zeros(N,1); % lower bound for alpha
    ub = ones(N,1)*C; % upper bound for alpha

    options = optimset('Display', 'off'); % turn off the annoying warnings from quadprog
    alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options); % solve for the alphas
    
    % compute the weights; w = SUM[ alpha_i * y_i * x_i ]
    w = zeros(1,D);
    for i = 1:N
        w = w + alpha(i,:)*Y_train(i,:)*X_train(i,:);
    end
    
    % calculate the bias for w^t*x + b
    
    AlmostZero=(abs(alpha)<max(abs(alpha))/1e5);
    alpha(AlmostZero)=0;
    S=find(alpha>0 & alpha<C);
    b = mean(Y_train(S)-X_train(S,:)*w');
   
    % predict the class of the test samples
    y_pred = zeros(M,1);
    for i = 1:M
        y_pred(i,:) = sign(dot(X_test(i,:),w) + b);
    end
end