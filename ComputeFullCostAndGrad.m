% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, trainingError, confusion ] = ComputeFullCostAndGrad( theta, decoder, data, hyperParams, ~ )
% Compute gradient and cost with regularization over a set of examples
% for some parameters.

N = length(data);

argout = nargout;
if nargout > 3
    confusions = zeros(N, 2);
end

accumulatedCost = 0;
accumulatedSuccess = 0;
if nargout > 1
    accumulatedGrad = zeros(length(theta), 1);
end

totalDiff = 0;
items = 0;

% Parallelize
if matlabpool('size') == 0 % checking to see if my pool is already open
    matlabpool;
end

if nargout > 1
    parfor i = 1:N
        [localCost, localGrad, localPred] = ...
            ComputeCostAndGrad(theta, decoder, data(i), hyperParams);
        accumulatedCost = accumulatedCost + localCost;
        accumulatedGrad = accumulatedGrad + localGrad;
        
        localCorrect = localPred == data(i).relation;
        ratingDifference = abs( (data(i).relation) - (localPred) );
%         disp('Correct');
%         disp(data(i).relation);
%         disp('Pred');
%         disp(localPred);
%         disp('diff');
%         disp(ratingDifference);
%         disp('-----------');
        if (~localCorrect) && (argout > 2) && hyperParams.showExamples
            disp(['for: ', data(i).leftTree.getText, ' ', ...
                  hyperParams.relations{data(i).relation}, ' ', ... 
            	  data(i).rightTree.getText, ...
                  ' h:  ', hyperParams.relations{localPred}]);
        end
        
        if argout > 3
            confusions(i,:) = [localPred, data(i).relation];
        end
        
        totalDiff = totalDiff + ratingDifference^2;
        items = items + 1;
        accumulatedSuccess = accumulatedSuccess + localCorrect;
    end
    
%      disp('totalDiff');
%      disp(totalDiff);
    average = sqrt(totalDiff / N);
     disp('average Euclid Dist');
     disp(average);
    
    if nargout > 3
        confusion = zeros(hyperParams.numRelations);
        for i = 1:N
           confusion(confusions(i,1), confusions(i,2)) = ...
               confusion(confusions(i,1), confusions(i,2)) + 1;
        end
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

% cost = [combinedCost normalizedCost regCost]; 
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
    trainingErrorSuccessRate = (accumulatedSuccess / N);
    disp('Percent Guess Exactly Right');
    disp(trainingErrorSuccessRate);
    disp('# Guessed Correctly');
    disp(accumulatedSuccess);
    disp('Out of:');
    disp(N);
    trainingError = average;
end

end