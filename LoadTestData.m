function LoadTestData(varsFilename, outFilename)

vars = {'theta','thetaDecoder'};
load(varsFilename,'-mat',vars{:});

[wordMap, relationMap, relations] = LoadTrainingData('./data-supp/vocab.csv');
[hyperParams, options] = LoadOptions(relations, '.', false, true);

trainListing = dir('data/advadj_train.csv');
testListing = dir('data/advadj_test.csv');
splitFilenames = {};
testFilenames = {testListing.name}
trainFilenames = {trainListing.name};
[~,testDatasets] = LoadConstitDatasets(trainFilenames,splitFilenames,...
				       testFilenames, wordMap, ...
				       relationMap, hyperParams);

disp('Running tests...')
fid = fopen(outFilename, 'w');

for i = 1:length(testDatasets{1})
  data = testDatasets{2}{i};
  N = length(data);
  for j=1:N
    [~,~,localPredDist] = ComputeCostAndGrad(theta, thetaDecoder, ...
					     data(j), hyperParams);
    localError = sum(data(j).relDist .* log(data(j).relDist ./ ...
					    localPredDist));
    adv = data(j).leftTree.getText();
    adj = data(j).rightTree.getText();
    
    outstr = strcat(adv, ',', adj, ',', num2str(localError));
    for k=1:10
	outstr = strcat(outstr, ',', num2str(data(j).relDist(k)));
    end
    for k = 1:10
	outstr = strcat(outstr, ',', num2str(localPredDist(k)));
    end
    outstr = strcat(outstr, '\n');
    fprintf(fid, outstr);
  end
end

fclose(fid);
end
