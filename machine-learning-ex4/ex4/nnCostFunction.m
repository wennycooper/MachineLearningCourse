function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% [KK] compute a1, add ones to the a1
a1 = [ones(m, 1) X];    % 5000x401

% [KK] compute z2, a2, and transpose a2 to the normal form as a1
z2 = Theta1*a1';       % 25x5000
a2 = sigmoid(z2);      % 25x5000
a2 = a2';              % 5000x25

% Add ones to the a2
a2 = [ones(m,1) a2];  % 5000x26

% compute z3, a3, and transpose a3
z3 = Theta2*a2';      % 10x5000
a3 = sigmoid(z3);     % 10x5000
a3 = a3';             % 5000x10

% [KK] This is in two for-loop form
%{
J = 0;
for i=1:m,
    yi = zeros(num_labels, 1);   % y is 10x1 binary vector
    yi(y(i)) = 1;                % only index y(i) is 1
    for k=1:num_labels,
        J = J + (-yi(k)*log(a3(i,k)) - (1-yi(k))*log(1-(a3(i,k))));
    end
end
J = J/m;
%}


% [KK] This is in vectorization form
J = 0;
for k=1:num_labels,
    J = J + sum(-(y==k)'*log(a3(:,k)) - (1-(y==k))'*log(1-(a3(:,k))));    
end
J = J/m;


% [KK] Compute Regularization part
JReg = 0;
for j=1:hidden_layer_size,
    for k=1:input_layer_size,
        JReg = JReg + (Theta1(j,k+1))^2;   % note that we need to skip the theta(k=1)
    end
end

for j=1:num_labels,
    for k=1:hidden_layer_size,
        JReg = JReg + (Theta2(j,k+1))^2;   % note that we need to skip the theta(k=1)
    end
end

% sum up
J = J + lambda/(2*m) * JReg;


% Backpropagation Update

triangle_1 = zeros(size(Theta1));
triangle_2 = zeros(size(Theta2));

for t=1:m,    
    % Step1
    a_1 = a1(t, :)';  % a_1 is column vector 401x1
    a_2 = a2(t, :)';  % a_2 is column vector 26x1
    
    % Step2
    yi = zeros(num_labels, 1);   % yi is 10x1 binary vector
    yi(y(t)) = 1;                % only yi(y(t)) is 1
    delta_3 = a3(t,:)' - yi;   % 10x1
    
    % Step3
    delta_2 = (Theta2)'*delta_3;    % 26x10 * 10x1 = 26x1
    delta_2 = delta_2(2:end);       % remove bias in delta_2, so that delta_2 is 25x1
    delta_2 = delta_2 .* sigmoidGradient(z2(:,t)); % 25x1 .* 25x1 = 25x1
    
    % Step4
    triangle_1 = triangle_1 + delta_2*(a_1)'; % 25x1 * 1x401 = 25x401
    triangle_2 = triangle_2 + delta_3*(a_2)'; % 10x1 * 1x26  = 10x26
    
end

% Step5

Theta1Temp = Theta1;
Theta1Temp(:,1) = zeros(hidden_layer_size,1);
Theta2Temp = Theta2;
Theta2Temp(:,1) = zeros(num_labels,1);



Theta1_grad = 1/m * triangle_1 + lambda/m * Theta1Temp;  % 25x401
Theta2_grad = 1/m * triangle_2 + lambda/m * Theta2Temp;  % 10x26



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
