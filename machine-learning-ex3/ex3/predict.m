function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
a1 = [ones(m, 1) X];   % 5000x401

% compute z2, a2, and transpose a2 to the normal form as a1
z2 = Theta1*a1';       % 25x5000
a2 = sigmoid(z2);      % 25x5000
a2 = a2';              % 5000x25

% Add ones to the a2
a2 = [ones(m,1) a2];  % 5000x26

% compute z3, a3, and transpose a3
z3 = Theta2*a2';      % 10x5000
a3 = sigmoid(z3);     % 10x5000
a3 = a3';             % 5000x10

[M,p] = max(a3, [], 2); % to obtain the max for each row and output the index to p



% =========================================================================


end
