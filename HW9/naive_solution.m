function prob = naive_solution(A, B, N, M, sent)
    states = permn(1:N, M); % create permutations of all possible states of length M (number of words in our sentence)
    numStates = size(states, 1); % number of states generated
    
    prob = 0.0;
    for i = 1:numStates % iterate over every possible state combination for our words
        % calculate the probability of this word/state combination occuring and add it to our total probability
        prob = prob + calc_sent_prob(sent, states(i, :), A, B);
    end
end

function prob = calc_sent_prob(sent, states, A, B)
    prob = 1.0;
    numWords = size(sent, 2);
    
    for i = 1:numWords % for each word in our sentence
        word = sent(:, i); % current word
        state = states(:, i); % current state
        if i == 1 % we are going from the start state to our first word
            prob = prob * B(state, word) * A(1, state+1);
        elseif i == numWords % we are going from the current state to the end state
            prevState = states(:, i-1);
            prob = prob * B(state, word) * A(prevState+1, state+1);
            prob = prob * A(state+1, 6); % traversing to end state
        else % all other possible states in between start and end
            prevState = states(:, i-1);
            prob = prob * B(state, word) * A(prevState+1, state+1);
        end
    end
end