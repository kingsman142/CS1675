% hmm_starter

N = 4; % number of states
M = 7; % words in vocabulary

vocab = {'john ', 'mary ', 'cat ', 'saw ', 'ate ', 'a ', 'the '};

A = zeros(N+2, N+2); % 1 start state + 4 pos state + 1 end state
% state 2 = PropNoun, state 3 = Noun, state 4 = Verb, state 5 = Det
A(1, :) = [0    0.4     0       0.1     0.5     0];
A(2, :) = [0    0       0       0.8     0.1     0.1];
A(3, :) = [0    0       0       0.85    0.05    0.1];
A(4, :) = [0    0.25    0       0       0.25    0.5];
A(5, :) = [0    0       0.95    0.05    0       0];

for i = 1:(N+1)
    assert(sum(A(i, :)) == 1);
end

B = zeros(N, M); % note we skip start/end states here so B(1, :) correponds to A(2, :)
B(1, :) = [0.4  0.4     0.1     0.01    0.05    0.03    0.01];
B(2, :) = [0.25 0.05    0.30    0.25    0.05    0.05    0.05];
B(3, :) = [0.04 0.05    0.04    0.45    0.4     0.01    0.01];
B(4, :) = [0.01 0.01    0.01    0.01    0.01    0.45    0.50];

for i = 1:N
    assert(sum(B(i, :)) == 1);
end

