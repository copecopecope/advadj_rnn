% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [softmaxGradient, softmaxDelta] = ...
    ComputeSoftmaxGradient (hyperParams, classifierParameters, ...
                            relationProbs, trueRelation, tensorOutput)
% Compute the gradient for the softmax layer parameters, and the deltas to
% pass down.
                        
softmaxGradient = zeros(size(classifierParameters, 1), ...
    hyperParams.penultDim + 1);

% Compute node softmax error
targetRelationProbs = zeros(length(relationProbs), 1);
targetRelationProbs(trueRelation) = 1;
softmaxDeltaFirstHalf = classifierParameters' * ...
                        (relationProbs - targetRelationProbs);
                    
% Compute nonlinearity and append intercept
softmaxDeltaSecondHalf = hyperParams.classNLDeriv([1; tensorOutput]);
softmaxDelta = (softmaxDeltaFirstHalf .* softmaxDeltaSecondHalf);

for relEval = 1:size(classifierParameters, 1)
    % Del from UFLDL Wiki on softmax
    softmaxGradient(relEval, :) = -([1; tensorOutput] .* ...
        ((trueRelation == relEval) - relationProbs(relEval)))';
end

softmaxDelta = softmaxDelta(2:hyperParams.penultDim+1);

end