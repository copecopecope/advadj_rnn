% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [combined, aggConfusion] = TestModel(CostGradFunc, theta, thetaDecoder, testDatasets, hyperParams)

% Evaluate on test datasets, and show set-by-set results while aggregating
% an overall confusion matrix.
aggConfusion = zeros(hyperParams.numRelations);
heldOutConfusion = zeros(hyperParams.numRelations);
targetConfusion = zeros(hyperParams.numRelations);

for i = 1:length(testDatasets{1})
    [~, ~, err, confusion] = CostGradFunc(theta, thetaDecoder, testDatasets{2}{i}, hyperParams);
    if i == 1
        targetErr = err;
        disp('targetErr');
        disp(targetErr);
        targetConfusion = confusion;
    end
    if i < hyperParams.firstSplit
        heldOutConfusion = heldOutConfusion + confusion;
    end
    if hyperParams.showConfusions && err > 0
        disp(['For ', testDatasets{1}{i}, ': ', num2str(err)])
        disp('GT:  1     2     3     4     5     6     7     8     9     10')
        disp(confusion)
    end
    aggConfusion = aggConfusion + confusion;
end

% Compute error rate from aggregate confusion matrix
aggErr = 1 - sum(sum(eye(hyperParams.numRelations) .* aggConfusion)) / sum(sum(aggConfusion));    
heldOutErr = 1 - sum(sum(eye(hyperParams.numRelations) .* heldOutConfusion)) / sum(sum(heldOutConfusion));

MacroF1 = [GetMacroF1(targetConfusion), GetMacroF1(heldOutConfusion), GetMacroF1(aggConfusion)];

disp(MacroF1)

combined = [targetErr, heldOutErr, aggErr];

end