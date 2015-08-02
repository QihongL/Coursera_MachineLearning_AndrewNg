function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
% Preallocate resources
idx = zeros(size(X,1), 1);

% Loop over all examples
for i = 1 : size(X,1)

    % Computing for the square of difference
    difference = bsxfun(@minus, centroids, X(i,:) );
    diffsqr = sum(difference.^2,2);
    
    % Assign a cluster
    idx(i) = find(diffsqr == min(diffsqr), 1 , 'first');

end

end

