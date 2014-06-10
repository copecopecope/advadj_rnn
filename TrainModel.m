function TrainModel(expName)

% define expName
if ~exist(expName, 'dir')
  mkdir(expName);
end

% load wordMap
[wordMap, relationMap, relations] = LoadTrainingData('./data-supp/vocab.csv');

% load hyperparameters and options for minFunc/adaGrad
[hyperParams, options] = LoadOptions(relations, expName);

listing = dir('data/advadj.csv');
splitFilenames = {listing.name};
testFilenames = {}
trainFilenames = {}

splitFilenames = setdiff(splitFilenames, testFilenames);
hyperParams.firstSplit = 3;

% Randomly initialize.
[ theta, thetaDecoder ] = InitializeModel(size(wordMap, 1), hyperParams);

% Load training/test data
[trainDataset, testDatasets] = ...
    LoadConstitDatasets(trainFilenames, splitFilenames, ...
    testFilenames, wordMap, relationMap);
% trainDataset = Symmetrize(trainDataset);

% Train
disp('Training')
options.MaxFunEvals = 1000;
options.DerivativeCheck = 'on';
options.runName = 'tr';

if hyperParams.minFunc
    % Set up minfunc
    addpath('minFunc_2012/minFunc/')
    addpath('minFunc_2012/minFunc/compiled/')
    addpath('minFunc_2012/minFunc/mex/')
    addpath('minFunc_2012/autoDif/')

    theta = minFunc(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, hyperParams, testDatasets);
    % TODO: Forget metadata and repeat?
else
    theta = adaGradSGD(@ComputeFullCostAndGrad, theta, options, ...
        thetaDecoder, trainDataset, ...
        hyperParams, testDatasets);
end

% Done. Evaluate final model on training data.
% (Mid-run results are usually better.)
[~, ~, trAcc, trConfusion] = ComputeFullCostAndGrad(theta, ...
    thetaDecoder, trainDataset, hyperParams);

disp('Training confusion, PER: ')
disp('tr:  1     2     3     4     5     6     7     8     9     10')
disp(trConfusion)
disp(trAcc)

[teAcc, teConfusion] = TestModel(@ComputeFullCostAndGrad, theta, ...
				 thetaDecoder, testDatasets, hyperParams);

% Print results for all three full datasets
disp('Word pair confusion, PER: ')
disp('tr:  1     2     3     4     5     6     7     8     9     10')
%disp(preConfusion)
%disp(preAcc)

disp('Training confusion, PER: ')
disp('tr:  1     2     3     4     5     6     7     8     9     10')
disp(trConfusion)
disp(trAcc)
disp(trAcc)

disp('Test confusion, PER: ')
disp('tr:  1     2     3     4     5     6     7     8     9     10')
disp(teConfusion)
disp(teAcc)

end
