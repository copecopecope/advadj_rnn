% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function o = Objective(trueRelation, relationProbs)
% Compute the non-regularized objective for a single example.

% disp('+');
% disp(trueRelation);
% disp(relationProbs);
% disp('-');
o = -log(relationProbs(trueRelation));

end