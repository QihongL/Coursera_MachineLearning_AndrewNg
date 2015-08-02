function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Set some parameters 
C_all = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_all = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% Preallocate
error = zeros(8,8);

% loop over C
for i = 1:8
    C = C_all(i);
    % loop over sigma
    for j = 1:8
        sigma = sigma_all(j);
        
        % Fit model
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        % Get the error 
        error(i,j) = mean(double(predictions ~= yval));
    end
end

% Get the min error
indice = find(error == min(error(:)), 1, 'first');
column = ceil(indice / 8);
row = mod(indice , 8);

% Find the corresponding C & sigma
C = C_all(row);
sigma = sigma_all(column);

end
