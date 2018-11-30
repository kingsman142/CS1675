function [y_pred, Z] = forward(X, W1, W2)
    % dot(...) multiplies weights by neuron values and tanh(...) 'standardizes' the output of all the neurons
    Z = tanh( activate_neurons(X, W1) );
    y_pred = dot(W2, Z);
end

% For a given hidden neuron, compute its activation value based on the
% previous layer's neurons' values and the weights connected between them
function activations = activate_neurons(X, W)
    M = size(W, 1);
    activations = zeros(M, 1);
    for i = 1:M % for each hidden neuron
        activations(i, :) = dot(X, W(i, :)); % activation = sum of (previous layer's neurons * weights conncted to this hidden neuron)
    end
end