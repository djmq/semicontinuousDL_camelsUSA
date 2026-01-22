% hurdle_training_pipeline.m
%
% Description:
%   Train, validate, and test a hurdle-LSTM streamflow model using
%   preprocessed CAMELS inputs/targets. The workflow:
%     - set paths and load preprocessed data
%     - configure training and model hyperparameters
%     - prepare dlarray/gpuArray inputs (and per-catchment stds)
%     - initialize hurdle model parameters (classification + regression)
%     - run training loop with Adam, gradient clipping, optional plot
%     - select best epoch by validation loss (uses per-catchment std)
%     - computes distributional outputs, derives median predictions, back-transforms to original scale 
%     - evaluates the best model on the test set and computes per-catchment metrics
%     - save checkpointed models and summary metrics to a MAT-file
%
% Usage:
%   Place this file in the project folder (or update paths at top),
%   ensure the required .mat files exist in the configured data folder,
%   then run:
%     hurdle_training_pipeline
%
% Required Input Files (loaded by script):
%   camels_421_partitioning.mat
%   camels_421_inputs.mat
%   camels_421_target_hurdle_rg.mat
%
% Primary Outputs (saved at end):
%   A MAT-file named hurdle_<q>_<nhid>_<epoch>_<miniBatchSize>_... .mat
%   containing:
%     - models           : checkpointed parameter structs by epoch
%     - D                : (placeholder; include if defined)
%     - valLoss          : validation loss per epoch
%     - bestModelInd     : index of selected best epoch
%     - results_median   : aggregated median test metrics
%     - results_mean     : aggregated mean test metrics
%     - q                : Burr / hurdle configuration parameter
%
% Required Helper Functions (must be on MATLAB path):
%   initHurdle
%   gradientsHurdle
%   forwardHurdle
%   predictionHurdle
%   distributionalParametersHurdle
%   quantileHurdle
%   lossHurdle
%   zerosLikeParams
%   modelGradientsCheck
%   thresholdL2Norm
%   camels_metrics
%
% Notes and Assumptions:
%   - Uses dlarray and gpuArray when a supported GPU is available.
%   - Random seed is set near the top; gpurng(seed) is used for GPU RNG.
%   - Expects variables from loaded .mat files:
%       trainIndices, catchmentPartitionInds, inputsScaled, targetScaled,
%       targetScaledStd, scaleTarget, offsetTarget, nc, etc.
%   - Hurdle model outputs classification (zero vs >0) and Burr XII
%     regression parameters (a,b,c). The script computes quantiles
%     (median by default) for deterministic evaluation.
%   - Validation and test batching assume contiguous blocks per catchment
%     as defined in catchmentPartitionInds.
%   - Saved filename encodes q, nhid, epoch, miniBatchSize, miniBatchFactor, seed.
%     Confirm 'D' exists if it should be saved.
%
% Example:
%   % From project folder (with data in data/ and helpers on path):
%   hurdle_training_pipeline
%
% Author:
%   John Quilty
%   Affiliation: University of Waterloo
%
% Contributors:
%   Mohammad Sina Jahangir (Reviewer)
%
% Date:
%   2026-01-21
%
% Version:
%   1.0
%
% License:
%   MIT License
%   Copyright (c) 2026 John Quilty.
%   Refer to LICENSE file distributed with the project.

%% Paths and data loading

% paths for data and code
addpath F:\projects\Paper_HYDROL74788\hurdle_DL_camels\hurdle\
addpath F:\projects\Paper_HYDROL74788\hurdle_DL_camels\data\
addpath F:\projects\Paper_HYDROL74788\hurdle_DL_camels\utilities\

% load in data (already pre-processed, ready to use in models)
cd F:\projects\Paper_HYDROL74788\hurdle_DL_camels\data\
load camels_421_partitioning
load camels_421_inputs
load camels_421_target_hurdle_rg

cd F:\projects\Paper_HYDROL74788\hurdle_DL_camels\hurdle\

%% Settings

% set seed for reproducibility
seed = 1; % also used to randomize initialization of model parameters
rng('default');
rng(seed); % reproducibility

% training
numEpochs = 30;
learningRate = 0.001;
miniBatchSize = 2048; % adjust as needed; however, each batch should contain "enough" samples where target=0
miniBatchFactor = 1; % how many batches to (pre-)load at a time
gradientThresholdMethod = "global-l2norm"; % gradient clipping
gradientThreshold = 2;

% lstm
nhid = 256; % number of hidden neurons in LSTM layer
seq_len = 365; % sequence length
dropoutRate = 0.4; % dropout rate after LSTM layer

% hurdle output layer
q = 3; % parameter that determines how many (finite) moments are required in Burr XII distribution
numResponsesRegression = 3; % Burr XII distribution has three parameters (a,b,c) ** do not change
numResponsesClassification = 2; % classifies if target > 0 or not ** do not change

% plotting
plots = "none"; %"none" OR "training-progress";
% plots = "training-progress";
verbose = true;
verboseFrequency = 50; %floor(7704*421/miniBatchSize);

%% Prepare data for model

% check if GPU is supported
checkGPU = canUseGPU;

% move data to GPU if available (model will be moved to GPU automatically
% after first iteration)

if checkGPU
    disp("GPU is available and supported.");

    gpurng('default');
    gpurng(seed); % ensure reproducibility on GPU

    inputsScaled = dlarray(single(inputsScaled),'BC');
    inputsScaled = gpuArray(inputsScaled);
    targetScaled = dlarray(single(targetScaled),'BC');
    targetScaled = gpuArray(targetScaled);

    repm = gpuArray(repmat( ...
        seq_len-1:-1:0,miniBatchFactor*miniBatchSize,1)).'; % helps create batch inputs more easily

else
    disp("No supported GPU available. Using CPU.");

    inputsScaled = dlarray(single(inputsScaled),'BC');
    targetScaled = dlarray(single(targetScaled),'BC');

    repm = repmat(seq_len-1:-1:0,miniBatchFactor*miniBatchSize,1).'; % helps create batch inputs more easily

end 

%% Initialize model

numFeatures = size(inputsScaled,1);
parameters = initHurdle(numResponsesRegression...
    , numResponsesClassification, numFeatures, nhid);


%% Training loop

% accelerate estimation of gradients
accfun = dlaccelerate(@gradientsHurdle);  
clearCache(accfun); % Clear any previously cached traces of the accelerated function using the clearCache function

% initialize Adam moments (same struct fields, same shapes)
trailingAvg   = zerosLikeParams(parameters);
trailingAvgSq = zerosLikeParams(parameters);

% if required, initialize the training progress plot 
if plots == "training-progress"
    figure
    lineLossTrain = animatedline(Color=[0.85 0.325 0.098]);
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end

ntr = numel(trainIndices);
numIterations = floor(ntr/miniBatchSize); % the number of iteration per one epoch
numSubEpochs = floor(ntr/(miniBatchSize*miniBatchFactor)); % the number of sub-epochs due to loadIterations

gradientsLast = [];
parametersLast = parameters;
models = struct('networks', cell(1, numEpochs)); % checkpoint models every epoch

disp("|======================================================================================================================|")
disp("|  Epoch  |  Iteration  |  Time Elapsed  |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  |  Base Learning  |")
disp("|         |             |   (hh:mm:ss)   |   Accuracy   |   Accuracy   |     Loss     |     Loss     |      Rate       |")
disp("|======================================================================================================================|")

iteration = 0;
gradientsFlagCount = 0;
start = tic;

% loop over epochs
for epoch = 1:numEpochs

    % randomly shuffle train indices
    shuffleTrainIndices = trainIndices(randperm(ntr));

    % move to GPU if available
    if checkGPU; shuffleTrainIndices = gpuArray(shuffleTrainIndices); end

    for subEpoch = 1:numSubEpochs

        % pre-load several minibatches then loop over them
        subTrainIndices = shuffleTrainIndices(...
            (subEpoch-1)*miniBatchFactor*miniBatchSize+1:...
            subEpoch*miniBatchFactor*miniBatchSize);

        rmm = subTrainIndices - repm;

        X = reshape(inputsScaled(:,rmm(:)), numFeatures, []...
            , miniBatchFactor*miniBatchSize);
        X = dlarray(X,'CTB');

        T = targetScaled(subTrainIndices); % batch target [CB]

        for i = 1:miniBatchFactor

            iteration = iteration + 1;

            newInds = (i-1)*miniBatchSize+1:i*miniBatchSize;

            % Evaluate the model loss, gradients, and state using dlfeval
            % and the accelerated function.
            [loss,gradients] = dlfeval(accfun...
                , parameters...
                , X(:,newInds,:)...
                , T(:,newInds)...
                , dropoutRate ...
                , q);

            % check for any issues updating gradients
            gradientsFlag = modelGradientsCheck(gradients);

            if gradientsFlag % issues w/ gradients; skip update, reduce iteration count, and move to next batch

                % warning("NaN in gradients at iteration %d", iteration);
                trailingAvg   = zerosLikeParams(parameters); 
                trailingAvgSq = zerosLikeParams(parameters);
                % learningRate = learningRate * 0.999; % reduce LR if desired
                parameters = parametersLast;

                gradients = gradientsLast;
                iteration = iteration - 1;
                gradientsFlagCount = gradientsFlagCount + 1;

            else % no issue with gradients; proceed to update parameters

                gradients = dlupdate(@(g) thresholdL2Norm(g...
                    , gradientThreshold),...
                    gradients);

                % update the network parameters using the Adam optimizer
                [parameters,trailingAvg,trailingAvgSq] = adamupdate(...
                    parameters...
                    , gradients ...
                    , trailingAvg, trailingAvgSq ...
                    , iteration ...
                    , learningRate);

                gradientsLast = gradients;
                parametersLast = parameters;

            end


            % display the training progress
            if plots == "training-progress"
                D = duration(0,0,toc(start),Format="hh:mm:ss");
                loss = double(loss);
                addpoints(lineLossTrain,iteration,loss)
                title("Epoch: " + epoch + ", Elapsed: " + string(D))
                drawnow
            end

            if verbose && (iteration == 1 || ...
                    mod(iteration,verboseFrequency) == 0)
                D = duration(0,0,toc(start),'Format','hh:mm:ss');


                % add if desired
                mseTrain = "";
                mseValidation = "";
                lossValidation = "";

                disp("| " + ...
                    pad(string(epoch),7,'left') + " | " + ...
                    pad(string(iteration),11,'left') + " | " + ...
                    pad(string(D),14,'left') + " | " + ...
                    pad(string(mseTrain),12,'left') + " | " + ...
                    pad(string(mseValidation),12,'left') + " | " + ...
                    pad(string(extractdata(loss)),12,'left') + " | " + ...
                    pad(string(lossValidation),12,'left') + " | " + ...
                    pad(string(learningRate),15,'left') + " |")


            end

        end

    end

    models(epoch).networks = parameters;


end

disp("|======================================================================================================================|")

disp("Training finished: Max Epochs Met.")


%% Prepare validation and test batches

% prepare indices to read batches for validation and test sets
iv = [];
it = [];
for i = 1:1:nc

    iv = [iv,catchmentPartitionInds(i,2) + 1:catchmentPartitionInds(i,3)]; % validation records
    it = [it, catchmentPartitionInds(i,3) + 1:catchmentPartitionInds(i,4)]; % test records

end

ivFullBatches = floor(numel(iv)/miniBatchSize);
ivRemainder = numel(iv) - miniBatchSize*ivFullBatches;
ivBatchesInds = cumsum(repelem(miniBatchSize,1,ivFullBatches));
ivBatchesInds = [...
    [1, ivBatchesInds(1:end-1)+1]; ivBatchesInds];

if ivRemainder > 0
    ivBatchesInds = [ivBatchesInds,...
        [ivBatchesInds(2,end)+1; ivBatchesInds(2,end)+ivRemainder]];

end
ivBatchesN = size(ivBatchesInds,2);

itFullBatches = floor(numel(it)/miniBatchSize);
itRemainder = numel(it) - miniBatchSize*itFullBatches;
itBatchesInds = cumsum(repelem(miniBatchSize,1,itFullBatches));
itBatchesInds = [...
    [1, itBatchesInds(1:end-1)+1]; itBatchesInds];

if itRemainder > 0
    itBatchesInds = [itBatchesInds,...
        [itBatchesInds(2,end)+1; itBatchesInds(2,end)+itRemainder]];

end
itBatchesN = size(itBatchesInds,2);

%% Identify "best" epoch/model using validation set

accfun2 = dlaccelerate(@forwardHurdle);
clearCache(accfun2) % clear any previously cached traces of the accelerated function using the clearCache function

numModels = numel(models);
valLoss = zeros(numModels,1);
Pred1 = dlarray(single(zeros(numResponsesRegression,numel(iv))),'CB');
Pred2 = dlarray(single(zeros(numResponsesClassification,numel(iv))),'CB');
if checkGPU; Pred1 = gpuArray(Pred1); end
if checkGPU; Pred2 = gpuArray(Pred2); end
tStart = tic;

validTargets_t = targetScaled(iv);

for ii = 1:numModels

    for iii = 1:ivBatchesN

        ivB = ivBatchesInds(1,iii):ivBatchesInds(2,iii);
        indsV = iv(ivB); % validation indices for batch
        miniBatchSizeV = ivBatchesInds(2,iii) - ivBatchesInds(1,iii) + 1; % in case of partial batch
        repmV = repmat(seq_len-1:-1:0,miniBatchSizeV,1).';
        indsVrmm = indsV - repmV;

        X = reshape(inputsScaled(:,indsVrmm(:)), numFeatures...
            , [], miniBatchSizeV);
        X = dlarray(X,'CTB');

        [Pred1(:,ivB), Pred2(:,ivB)] = predictionHurdle(accfun2 ...
            , models(ii).networks, X, dropoutRate);

    end
    
    valLoss(ii) = lossHurdle(Pred1, Pred2, validTargets_t, q);


    disp("Validation loss for epoch: "+string(ii) + ".....is: "+...
        string(valLoss(ii))+".....taken: "+toc(tStart)+ "   seconds...");
    
end
[~,bestModelInd] = min(valLoss());
bestParameters = models(bestModelInd).networks;

%% Estimate out-of-sample performance using test set

clearCache(accfun2)

% Test best model

Pred1 = dlarray(single(zeros(numResponsesRegression,numel(it))),'CB');
Pred2 = dlarray(single(zeros(numResponsesClassification,numel(it))),'CB');
if checkGPU; Pred1 = gpuArray(Pred1); end
if checkGPU; Pred2 = gpuArray(Pred2); end
for iii = 1:itBatchesN

    itB = itBatchesInds(1,iii):itBatchesInds(2,iii);
    indsT = it(itB); % test indices for batch
    miniBatchSizeT = itBatchesInds(2,iii) - itBatchesInds(1,iii) + 1; % in case of partial batch

    repmT = repmat(seq_len-1:-1:0,miniBatchSizeT,1).';
    indsTrmm = indsT - repmT;


    X = reshape(inputsScaled(:,indsTrmm(:)), numFeatures ...
        , [], miniBatchSizeT);
    X = dlarray(X,'CTB');

    [Pred1(:,itB), Pred2(:,itB)] = predictionHurdle(accfun2 ...
        , bestParameters, X, dropoutRate);

end

% transform hurdle predictions into distributional parameters

[b_t, c_t, a_t, prob0_t]  = distributionalParametersHurdle(Pred1, Pred2 ...
    , q);

% estimate median of hurdle distribution (deterministic prediction)
pmed = 0.5; % median (pmed=0.5)
q_t = quantileHurdle(b_t, c_t, a_t, prob0_t(2,:), pmed); 

% other quantiles calculated in the same way...
% p99 = 0.99; % use for .99 quantile
% q_t_99 = quantileHurdle(b_t, c_t, a_t, prob0_t(2,:), p99); 

% back-transform predictions and target to original scale
testPredictions_bt = (gather(extractdata(q_t)) .* ...
    scaleTarget + offsetTarget).';

testTargets_bt = gather(extractdata(targetScaled(it).* ...
    scaleTarget + offsetTarget)).';



% calculate performance metrics for each catchment across test set

% test set indices for first catchment
i_it = 1:catchmentPartitionInds(1,4) - catchmentPartitionInds(1,3);

nse_c = zeros(nc,1);
kge_c = zeros(nc,1);
nrmse_c = zeros(nc,1);
tsfhvrmse_c = zeros(nc,1);
tsflvrmse_c = zeros(nc,1);
tsfmsrmse_c = zeros(nc,1);

for iii = 1:nc

    iTestPredictions = testPredictions_bt(i_it);
    iTestTarget = testTargets_bt(i_it);
    cmetrics =  camels_metrics(iTestTarget, iTestPredictions);
    nse_c(iii) = cmetrics.nse;
    kge_c(iii) = cmetrics.kgem;
    nrmse_c(iii) = cmetrics.nrmse;
    tsfhvrmse_c(iii) = cmetrics.thrmse;
    tsflvrmse_c(iii) = cmetrics.tlrmse;
    tsfmsrmse_c(iii) = cmetrics.tmrmse;


    % test set indices for next catchment
    if iii < nc 

        i_it = i_it(end) + 1:...
            i_it(end) + catchmentPartitionInds(iii+1,4) -...
            catchmentPartitionInds(iii+1,3);        

    end


end

results_median = [
    median(nse_c); 
    median(kge_c); 
    median(tsflvrmse_c); 
    median(tsfhvrmse_c); 
    median(tsfmsrmse_c); 
    median(nrmse_c); 
    ];

results_mean = [ 
    mean(nse_c); 
    mean(kge_c); 
    mean(tsflvrmse_c); 
    mean(tsfhvrmse_c); 
    mean(tsfmsrmse_c); 
    mean(nrmse_c); 
    ];

%% Save models and summary of results

save_name=strcat('hurdle_',...
    num2str(q), '_', ...
    num2str(nhid),'_',...
    num2str(epoch),'_',...
    num2str(miniBatchSize),"_",...
    num2str(miniBatchFactor),"_",...
    num2str(seed),'.mat');
save(save_name...
    ,'models','D','valLoss','bestModelInd' ...
    , 'results_median', 'results_mean', 'q')
