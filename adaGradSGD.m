% Want to distribute this code? Have other questions? -> sbowman@stanford.edu
function [ theta ] = adaGradSGD(CostGradFunc, theta, options, thetaDecoder, trainingData, ...
    hyperParams, testDatasets)
% Home-baked implementation of SGD with AdaGrad.

N = length(trainingData);
prevCost = intmax;
bestTestErr = 1;
lr = options.lr;
sumSqGrad = zeros(size(theta));



for pass = 0:options.numPasses - 1

    fid_err = fopen([options.name, '/', 'errLog.csv'], 'a');
    fid_dists = fopen([options.name, '/dists@', num2str(pass), '.csv'], 'a');
        
    % Test on training data
    if mod(pass, options.examplesFreq) == 0 && pass > 0
        hyperParams.showExamples = true;
        disp('Training data:')
    else
        hyperParams.showExamples = false;
    end
    [cost, ~, acc] = CostGradFunc(theta, thetaDecoder, trainingData, hyperParams);
    
    % Test on test data
    if nargin > 5
        [testErr,~,dists] = TestModel(CostGradFunc, theta, thetaDecoder, testDatasets, hyperParams);
        bestTestErr = min(testErr, bestTestErr);
    else
        testErr = -1;
    end
    if testErr ~= -1
        disp(['pass ', num2str(pass), ' train KL: ', num2str(acc), ...
              ' test KL: ', num2str(testErr), ' (best: ', ...
              num2str(bestTestErr), ')']);
        fprintf(fid_err, strcat([num2str(pass), ',', num2str(acc), ',',num2str(testErr), '\n']));
    else
        disp(['pass ', num2str(pass), ' KL: ', num2str(acc)]);
    end
      % save if results are best yet!
    if bestTestErr == testErr
        save([options.name, '/', 'theta-', options.runName, '@', ...
            num2str(pass), '#', num2str(bestTestErr,4)] , ...
            'theta', 'thetaDecoder');

         N = size(dists,1);
        for i = 1:N
            dist_str = num2str(dists(i,21)); % KL-div
            for j = 1:20
                dist_str = strcat(dist_str, ',', num2str(dists(i,j)));
            end
            dist_str = strcat(dist_str, '\n');
            fprintf(fid_dists, dist_str);
        end
    end

    disp(['pass ', num2str(pass), ' cost: ', num2str(cost)]);
    if abs(prevCost - cost(1)) < 10e-7
        disp('Stopped improving.');
        break;
    end
    prevCost = cost(1);

    numBatches = ceil(N/options.miniBatchSize);
    randomOrder = randperm(N);

    for batchNo = 0:(numBatches-1)
        beginMiniBatch = (batchNo * options.miniBatchSize+1);
        endMiniBatch = min((batchNo+1) * options.miniBatchSize,N);
        batchInd = randomOrder(beginMiniBatch:endMiniBatch);
        batch = trainingData(batchInd);
        [ ~, grad ] = CostGradFunc(theta, thetaDecoder, batch, hyperParams);
        sumSqGrad = sumSqGrad + grad.^2;

        % Do AdaGrad update
        adaEps = 0.001;
        theta = theta - lr * (grad ./ (sqrt(sumSqGrad) + adaEps));
    end

    if mod(pass + 1, options.resetSumSqFreq) == 0
        sumSqGrad = zeros(size(theta));
    end

    fclose(fid_err);
    fclose(fid_dists);
end

end
