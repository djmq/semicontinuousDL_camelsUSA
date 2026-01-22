% deterministic_training_pipeline.m
%
% Description:
%   Train, validate, and test a deterministic LSTM-based streamflow model
%   using preprocessed CAMELS inputs/targets. The script:
%     - sets paths and loads preprocessed data
%     - configures training / model hyperparameters
%     - prepares dlarray/gpuArray inputs
%     - initializes LSTM model parameters
%     - runs the training loop with Adam updates and optional training plot
%     - selects the best epoch by validation loss
%     - evaluates the best model on the test set and computes per-catchment metrics
%     - saves models and summary results to a MAT-file
%
% Usage:
%   Place this file in the 'deterministic' folder (or update paths below),
%   ensure required data files exist in the data folder, then run the script.
%
% Required Input Files (loaded by script):
%   camels_421_partitioning.mat
%   camels_421_inputs.mat
%   camels_421_target_deterministic.mat
%
% Primary Outputs (saved at end):
%   A MAT-file named deterministic_<nhid>_<epoch>_<miniBatchSize>_...
%   containing:
%     - models       : checkpointed parameter structs by epoch
%     - D            : (placeholder variable used in save; ensure defined)
%     - valLoss      : validation loss per epoch
%     - bestModelInd : index of selected best epoch
%     - results_median, results_mean : aggregated test metrics
%
% Required Helper Functions (must be on MATLAB path):
%   initializeInitialHiddenOrCellState
%   initD, gradientsD, forwardD, predictionD, lossD
%   initializeGlorot_v3, initializeOrthogonal_v3, initializeUnitForgetGate_v3
%   initializeZeros_v3, zerosLikeParams, modelGradientsCheck, 
%   thresholdL2Norm, camels_metrics
%
% Notes and Assumptions:
%   - This script uses dlarray and optionally gpuArray if a supported GPU is
%     available. If GPU is available, inputs/targets are moved to GPU.
%   - Random seed is set at the top for reproducibility; gpurng is used for
%     GPU RNG.
%   - Many parameters (numEpochs, nhid, miniBatchSize, dropoutRate, etc.)
%     are configured in the Settings section and can be tuned before running.
%   - The script expects workspace variables produced by the loaded .mat
%     files (trainIndices, catchmentPartitionInds, inputsScaled, targetScaled,
%     targetScaledStd, scaleTarget, offsetTarget, nc, etc.). Ensure these
%     variables exist after loading.
%   - The saved file name includes nhid, epoch, miniBatchSize, miniBatchFactor,
%     and seed. Confirm 'D' is defined if it is intended to be saved.
%
% Example:
%   % From folder containing this script (and with data in the configured
%   % data path), simply run:
%   deterministic_training_pipeline
%
% Author:
%   John Quilty (Primary Author)
%   Affiliation: University of Waterloo
%
% Contributors:
%   Mohammad Sina Jahangir   Reviewer (Method review)
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
addpath F:\projects\Paper_HYDROL74788\hurdle_DL_camels\deterministic\
addpath F:\projects\Paper_HYDROL74788\hurdle_DL_camels\data\
addpath F:\projects\Paper_HYDROL74788\hurdle_DL_camels\utilities\

% load in data (already pre-processed, ready to use in models)
cd F:\projects\Paper_HYDROL74788\hurdle_DL_camels\data\
load camels_421_partitioning
load camels_421_inputs
load camels_421_target_deterministic

cd F:\projects\Paper_HYDROL74788\hurdle_DL_camels\deterministic\

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

% output layer
numResponses = 1; % ** do not change

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
    targetScaledStd = dlarray(targetScaledStd,'BC'); % scaled std of streamflow in each catchment required for loss
    targetScaledStd = gpuArray(targetScaledStd);

    repm = gpuArray(repmat( ...
        seq_len-1:-1:0,miniBatchFactor*miniBatchSize,1)).'; % helps create batch inputs more easily

else
    disp("No supported GPU available. Using CPU.");

    inputsScaled = dlarray(single(inputsScaled),'BC');
    targetScaled = dlarray(single(targetScaled),'BC'); 
    targetScaledStd = dlarray(targetScaledStd,'BC'); % scaled std of streamflow in each catchment required for loss

    repm = repmat(seq_len-1:-1:0,miniBatchFactor*miniBatchSize,1).'; % helps create batch inputs more easily

end 

%% Initialize model

numFeatures = size(inputsScaled,1);
parameters = initD(numResponses, numFeatures, nhid);


%% Training loop

% accelerate estimation of gradients
accfun = dlaccelerate(@gradientsD);  
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
        TStd = targetScaledStd(subTrainIndices); % batch std of target for specific catchment [CB]

        for i = 1:miniBatchFactor

            iteration = iteration + 1;

            newInds = (i-1)*miniBatchSize+1:i*miniBatchSize;

            % Evaluate the model loss, gradients, and state using dlfeval
            % and the accelerated function.
            [loss,gradients] = dlfeval(accfun...
                , parameters...
                , X(:,newInds,:)...
                , T(:,newInds)...
                , TStd(:,newInds)...
                , dropoutRate);

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

accfun2 = dlaccelerate(@forwardD);
clearCache(accfun2) % clear any previously cached traces of the accelerated function using the clearCache function

numModels = numel(models);
valLoss = zeros(numModels,1);
Pred = dlarray(single(zeros(numResponses,numel(iv))),'CB');
if checkGPU; Pred = gpuArray(Pred); end

validTargets_t = targetScaled(iv);
validTargetsStd_t = targetScaledStd(iv);

tStart = tic;
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

        Pred(:,ivB) = predictionD(accfun2 ...
            , models(ii).networks, X, dropoutRate);

    end
    
    valLoss(ii) = lossD(Pred, validTargets_t, validTargetsStd_t);


    disp("Validation loss for epoch: "+string(ii) + ".....is: "+...
        string(valLoss(ii))+".....taken: "+toc(tStart)+ "   seconds...");
    
end
[~,bestModelInd] = min(valLoss());
bestParameters = models(bestModelInd).networks;

%% Estimate out-of-sample performance using test set

clearCache(accfun2)

% Test best model

Pred = dlarray(single(zeros(numResponses,numel(it))),'CB');
if checkGPU; Pred = gpuArray(Pred); end

for iii = 1:itBatchesN

    itB = itBatchesInds(1,iii):itBatchesInds(2,iii);
    indsT = it(itB); % test indices for batch
    miniBatchSizeT = itBatchesInds(2,iii) - itBatchesInds(1,iii) + 1; % in case of partial batch

    repmT = repmat(seq_len-1:-1:0,miniBatchSizeT,1).';
    indsTrmm = indsT - repmT;


    X = reshape(inputsScaled(:,indsTrmm(:)), numFeatures ...
        , [], miniBatchSizeT);
    X = dlarray(X,'CTB');

    Pred(:,itB) = predictionD(accfun2, bestParameters, X, dropoutRate);

end

% back-transform predictions and target to original scale
testPredictions_bt = (gather(extractdata(Pred)) .* ...
    scaleTarget + offsetTarget).';
testPredictions_bt = max(0, testPredictions_bt); % adjust for lower-bound of streamflow

testTargets_bt = gather(extractdata(targetScaled(it).* ...
    scaleTarget + offsetTarget)).';
testTargets_bt = max(0, testTargets_bt); % adjust for lower-bound of streamflow



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

save_name=strcat('deterministic_',...
    num2str(nhid),'_',...
    num2str(epoch),'_',...
    num2str(miniBatchSize),"_",...
    num2str(miniBatchFactor),"_",...
    num2str(seed),'.mat');
save(save_name...
    ,'models','D','valLoss','bestModelInd' ...
    , 'results_median', 'results_mean')
