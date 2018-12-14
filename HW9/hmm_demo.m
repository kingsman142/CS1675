vocab = {'john', 'mary', 'cat', 'saw', 'ate', 'a', 'the'}; % all the words in our network
N = 4; % number of states
M = 7; % number of words in vocabulary
sentence = "cat saw the john."; % sample sentence

% Transition Matrix (indexed by trans(previousState, currentState))
trans = zeros(N+2, N+2); % 1 start state + 4 pos state + 1 end state
% state 2 = PropNoun, state 3 = Noun, state 4 = Verb, state 5 = Det
trans(1, :) = [0    0.4     0       0.1     0.5     0];
trans(2, :) = [0    0       0       0.8     0.1     0.1];
trans(3, :) = [0    0       0       0.85    0.05    0.1];
trans(4, :) = [0    0.25    0       0       0.25    0.5];
trans(5, :) = [0    0       0.95    0.05    0       0];

% Observation Matrix (indexed by obs(partOfSpeech, wordNumber))
obs = zeros(N, M); % note we skip start/end states here so obs(1, :) correponds to trans(2, :)
obs(1, :) = [0.4  0.4     0.1     0.01    0.05    0.03    0.01];
obs(2, :) = [0.25 0.05    0.30    0.25    0.05    0.05    0.05];
obs(3, :) = [0.04 0.05    0.04    0.45    0.4     0.01    0.01];
obs(4, :) = [0.01 0.01    0.01    0.01    0.01    0.45    0.50];

sent = convertSentenceToNums(sentence, vocab); % convert words to numbers
M = size(sent, 2); % number of words in our sentence

prob = naive_solution(trans, obs, N, M, sent); % probability of the sample sentence occuring
fprintf("The probability of \'%s\' occuring is %.9f\n", sentence, prob);

function sent = convertSentenceToNums(sentence, vocab)
    words = strsplit(sentence, ' '); % split by whitespace
    numWords = size(words, 2);
    sent = zeros(1, numWords);
    
    for i = 1:numWords
        word = strsplit(words(:, i), '.'); % split by periods
        if size(word, 2) > 1 % if a period exists, there will be 2+ elements in the list; the actual word is in the 1st index
            word = word(:, 1);
        end
        sent(1, i) = convertWordToNum(word, vocab);
    end
end

% Given our known vocabulary, tell us what the number equivalent of each word is
function num = convertWordToNum(word, vocab)
    num = 1; % default value
    num = find(vocab == word);
end