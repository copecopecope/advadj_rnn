% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ cost, grad, trainingError, confusion, predDist ] = ComputeFullCostAndGrad( theta, decoder, data, hyperParams, ~ )
% Compute gradient and cost with regularization over a set of examples
% for some parameters.

N = length(data);
A = containers.Map;


% Regenerate Map
for i=2:N
  if mod(i,1000) == 0
     fprintf('.');
  end
  %adv = data(i).leftTree;
    advadj = strcat(data(i).leftText,',',data(i).rightText)
   
    if ~isKey(A,advadj)
       A(advadj) = zeros(1,10);
    end
end
     
allKeys = keys(A);
lenKeys = length(allKeys);

fprintf('Filtering out pairs with less than %d instances...\n', 1);
for i=1:lenKeys
  key = allKeys(i);
  key = key{1};
  numPairs = sum(A(key));
  if numPairs < 1
     remove(A,key);
  end
end

predDist = zeros(lenKeys,10);
map = containers.Map;
totalPairs = 0;
for i=1:lenKeys
  key = allKeys(i);
  key = key{1};
  map(key) = i;
end


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
    for i = 1:N
        [localCost, localGrad, localPred, localPredDist] = ...
            ComputeCostAndGrad(theta, decoder, data(i), hyperParams);
        accumulatedCost = accumulatedCost + localCost;
        accumulatedGrad = accumulatedGrad + localGrad;
        
       
        advadj = strcat(data(i).leftText,',',data(i).rightText);
        if isKey(map,advadj)
            currInd = map(advadj);
            distr = predDist(currInd,:);
             for k = 1:10
                distr(k) = distr(k) + localPredDist(k);
                % TODO: check to make sure we should not be replacing current
                % line rather than adding to it
             end
            predDist(currInd,:) = distr;
        end
    

        
        
            
        
        
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
