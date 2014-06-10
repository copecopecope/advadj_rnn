% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [aggErr, aggConfusion] = TestModel(CostGradFunc, theta, thetaDecoder, testDatasets, hyperParams)

% % Evaluate on test datasets, and show set-by-set results while aggregating
% % an overall confusion matrix.
aggConfusion = zeros(hyperParams.numRelations);
% heldOutConfusion = zeros(hyperParams.numRelations);
% targetConfusion = zeros(hyperParams.numRelations);

aggErr = 0;

for i = 1:length(testDatasets{1})
    [~, ~, err] = CostGradFunc(theta, thetaDecoder, testDatasets{2}{i}, hyperParams);
    aggErr = aggErr + err;
end

end