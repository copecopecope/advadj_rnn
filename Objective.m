% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function o = Objective(trueDist, predDist)
% Compute the non-regularized objective for a single example.

% o = -log(relationProbs(trueRelation));
% disp(trueDist)
% disp(predDist)

o = -1 * sum(trueDist .* log(predDist));

end