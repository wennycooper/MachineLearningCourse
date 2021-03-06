function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% [KK] computer regularizationPart

th2 = theta.^2;
th2(1) = 0;
regPart = lambda/(2*m) * sum(th2);

% [KK] compute J in vectorization form
J = 1/(2*m)*(X*theta-y)' * (X*theta-y) + regPart; 


% [KK] compute gradient
grad2ndPart = (lambda/m).*theta;
grad2ndPart(1) = 0;

grad = 1/m * X'*(X*theta - y) + grad2ndPart;


% =========================================================================

grad = grad(:);

end
