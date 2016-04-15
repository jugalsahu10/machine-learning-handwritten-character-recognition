function gradientOfSigmoidFunction = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z

gradientOfSigmoidFunction = zeros(size(z));

gradientOfSigmoidFunction = sigmoid(z) .* (1 - sigmoid(z));

end
