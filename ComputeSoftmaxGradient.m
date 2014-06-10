% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [softmaxGradient, softmaxDelta] = ...
    ComputeSoftmaxGradient (hyperParams, classifierParameters, ...
                            predDist, trueDist, tensorOutput)
% Compute the gradient for the softmax layer parameters, and the deltas to
% pass down.
                        
softmaxGradient = zeros(size(classifierParameters, 1), ...
    hyperParams.penultDim + 1);

% disp(size(classifierParameters))

% % Compute node softmax error
% softmaxDeltaFirstHalf = classifierParameters' * ...
%                         (predDist - trueDist);

% disp(size(softmaxDeltaFirstHalf))
                    
% % Compute nonlinearity and append intercept
% softmaxDeltaSecondHalf = hyperParams.classNLDeriv([1; tensorOutput]);
% softmaxDelta = (softmaxDeltaFirstHalf .* softmaxDeltaSecondHalf);

% for relEval = 1:size(classifierParameters, 1)
%     % Del from UFLDL Wiki on softmax
%     % softmaxGradient(relEval, :) = -([1; tensorOutput] .* ((trueRelation == relEval) - relationProbs(relEval)))';
% end



% softmaxDelta = softmaxDelta(2:hyperParams.penultDim+1);

% Borrows from Andrew Maas RNN code
% compute nonlinearity??

softmaxDelta = predDist - trueDist;
softmaxGradient(:,1) = softmaxDelta; % bias
softmaxGradient(:,2:hyperParams.penultDim+1) = softmaxDelta*tensorOutput';

softmaxDelta = classifierParameters(:,2:hyperParams.penultDim+1)'*softmaxDelta;

end