function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

%% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

    for i = 1:num_iters
        % Perform a single gradient step on the parameter vector theta.
        % update: theta <- theta - (learning rate)*(derivative of cost function)
        temp = theta - (alpha / m) * ( X' * (X * theta - y))
        theta = temp    

        % This is also right, of course
%         theta = theta - ((alpha / m) * (X * theta - y)' * X)'

        % Save the cost J in every iteration    
        J_history(i) = computeCost(X, y, theta);
        
    end
%     plot(J_history, 1:1500)
end



