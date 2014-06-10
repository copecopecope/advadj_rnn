% Want to distribute this code? Have other questions? ->
% sbowman@stanford.edu
function [ cost, grad, trainingError, confusion, dists ] = ComputeFullCostAndGrad( theta, decoder, data, hyperParams, ~ )
% Compute gradient and cost with regularization over a set of examples
% for some parameters.

N = length(data);

argout = nargout;
if nargout > 3
    confusion = zeros(N, 2);
end

accumulatedCost = 0;
accumulatedError = 0;
if nargout > 1
    accumulatedGrad = zeros(length(theta), 1);
end

dists = zeros(N,21);

% Parallelize
if matlabpool('size') == 0 % checking to see if my pool is already open
    matlabpool;
end

if nargout > 1
    parfor i = 1:N
        [localCost, localGrad, localPredDist] = ...
            ComputeCostAndGrad(theta, decoder, data(i), hyperParams);
        accumulatedCost = accumulatedCost + localCost;
        accumulatedGrad = accumulatedGrad + localGrad;
        
        localError = sum(data(i).relDist .* log(data(i).relDist ./ localPredDist)) % KL div

        % dists_ = zeros(N,21)
        % if hyperParams.recordError
        %     dists_(i,1:10) = localPredDist;
        %     dists_(i,11:20) = data(i).relDist;
        %     dists_(i,21) = localError;
        % end
        % dists = dists + dists_;

        accumulatedError = accumulatedError + localError;
    end
    
else
    parfor i = 1:N
        localCost = ...
            ComputeCostAndGrad(theta, decoder, data(i), hyperParams);
        accumulatedCost = accumulatedCost + localCost;
    end
end

% Take mean cost.
normalizedCost = (1/length(data) * accumulatedCost);

if hyperParams.norm == 2
    % Apply L2 regularization
    regCost = hyperParams.lambda/2 * sum(theta.^2);
else
    % Apply L1 regularization
    regCost = hyperParams.lambda * sum(abs(theta)); 
end
combinedCost = normalizedCost + regCost;

cost = combinedCost;

if nargout > 1
    grad = (1/length(data) * accumulatedGrad);
    if hyperParams.norm == 2
        % Apply L2 regularization
        grad = grad + hyperParams.lambda * theta;
    else
        % Apply L1 regularization
        grad = grad + hyperParams.lambda * sign(theta);
    end
    trainingError = accumulatedError/N;
end

end
