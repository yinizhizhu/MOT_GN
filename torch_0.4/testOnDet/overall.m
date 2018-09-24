clc;
clear;

addpath(genpath('.'));
benchmark = 'MOT16';
% Benchmark specific properties
world = 0;
threshold = 0.5;

basic_dir = '/home/lee/Desktop/MOT/Results/MOT16/IoU';
metrics_dir = 'MotMetrics/';
metrics = {'IDF1', 'IDP', 'IDR', 'Rcll', 'Prcn', 'FAR', 'GT', 'MT', 'PT', ...
    'ML', 'FP', 'FN', 'IDs', 'FM', 'MOTA', 'MOTP', 'MOTAL'};
datasets = {'balanced', 'balanced_FP', 'balancedNearby', 'balancedNearby_FP'};
types = {'training', 'validation'};

if ~exist(metrics_dir, 'dir')
    mkdir(metrics_dir);
    fprintf('There is no %s! And we create this dir\n',metrics_dir);
end

seqs = [2, 4, 5, 9, 10, 11, 13]; % the set of sequences


% the length of the training dataset
tts = [];
tts(1) = 100; % for the cross evaluation
tts(2) = 200;

for j=1:length(tts) %Iteration for training sequence
    tt = tts(j);

    motmetrics_dir = fullfile(metrics_dir, sprintf('%d.txt', tt));
    fout = fopen(motmetrics_dir, 'w');
    
    for data_select=1:length(datasets)
        dataset = datasets{data_select};

        for k=1:length(types)   %Iteration for different condition
            % Read sequence list
            type = types{k};
            gtMat = [];
            resMat = [];

            % Evaluate sequences individually
            allMets = [];
            metsBenchmark = [];
            metsMultiCam = [];

            fprintf(fout, '%d %s %s\n', tt, dataset, type);
            fprintf(fout, 'Sequence');
            for x=1:length(metrics)
                fprintf(fout, '\t%s', metrics{x});
            end
            fprintf(fout, '\n');


            ind = 0;
            for x=1:7
                % the current training dataset
%             disp(t_dir);
                seq_index = sprintf('%02d', seqs(x));
                t_dir = fullfile(basic_dir, sprintf('%s/%d/motmetrics_inner_%s_dets', seq_index, tt, dataset));

                % Output the result of motmetrics into the text file
                ind = ind+1;
                gt_training = fullfile(t_dir, sprintf('gt_%s.txt', type));
                res_training = fullfile(t_dir, sprintf('res_%s.txt', type));
                fprintf('%s, %s\n', gt_training, res_training);

                % Parse ground truth
                % MOTX parsing
                gtdata = dlmread(gt_training);
                gtdata(gtdata(:,7)==0,:) = [];     % ignore 0-marked GT
                gtdata(gtdata(:,1)<1,:) = [];      % ignore negative frames
                if strcmp(benchmark, 'MOT16') || strcmp(benchmark, 'MOT17')  % ignore non-pedestrians
                    gtdata(gtdata(:,8)~=1,:) = [];
                end

                [~, ~, ic] = unique(gtdata(:,2)); % normalize IDs
                gtdata(:,2) = ic;
                gtMat{ind} = gtdata;


                % Parse result
                % MOTX data format
    %             if strcmp(benchmark, 'MOT16') || strcmp(benchmark, 'MOT17')
    %                 disp(t_dir);
    %                 res_training = preprocessResult(res_training, 'gt_training', t_dir);
    %             end

                % Skip evaluation if output is missing
                if ~exist(res_training, 'file')
                    error('Invalid submission. Result for sequence %s not available!\n',seq_index);
                end

                % Read result file
                if exist(res_training,'file')
                    s = dir(res_training);
                    if s.bytes ~= 0
                        resdata = dlmread(res_training);
                    else
                        resdata = zeros(0,9);
                    end
                else
                    error('Invalid submission. Result file for sequence %s is missing or invalid\n', res_training);
                end
                resdata(resdata(:,1)<1,:) = [];      % ignore negative frames
                resdata(resdata(:,1) > max(gtMat{ind}(:,1)),:) = []; % clip result to gtMaxFrame
                resMat{ind} = resdata;


                % Sanity check
                frameIdPairs = resMat{ind}(:,1:2);
                [u,I,~] = unique(frameIdPairs, 'rows', 'first');
                hasDuplicates = size(u,1) < size(frameIdPairs,1);
                if hasDuplicates
                    ixDupRows = setdiff(1:size(frameIdPairs,1), I);
                    dupFrameIdExample = frameIdPairs(ixDupRows(1),:);
                    rows = find(ismember(frameIdPairs, dupFrameIdExample, 'rows'));

                    errorMessage = sprintf('Invalid submission: Found duplicate ID/Frame pairs in sequence %s.\nInstance:\n', sequenceName);
                    errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(1),:)), sprintf('\n')];
                    errorMessage = [errorMessage, sprintf('%10.2f', resMat{ind}(rows(2),:)), sprintf('\n')];
                    assert(~hasDuplicates, errorMessage);
                end

                % Evaluate sequence
                [metsCLEAR, mInf, additionalInfo] = CLEAR_MOT_HUN(gtMat{ind}, resMat{ind}, threshold, world);
                metsID = IDmeasures(gtMat{ind}, resMat{ind}, threshold, world);
                mets = [metsID.IDF1, metsID.IDP, metsID.IDR, metsCLEAR];
                allMets(ind).name = seq_index;
                allMets(ind).m    = mets;
                allMets(ind).IDmeasures = metsID;
                allMets(ind).additionalInfo = additionalInfo;
                fprintf('%s,%s,%s\n', seq_index, dataset, types{k}); printMetrics(mets);
                fprintf(fout, '%s', seq_index);
                for l=1:length(mets)
                    fprintf(fout,'\t%f', mets(l));
                end
                fprintf(fout, '\n');
            end
            % Overall scores
            metsBenchmark = evaluateBenchmark(allMets, world);
            fprintf(' ********************* Your %d %s Results *********************\n', tt, type);
            printMetrics(metsBenchmark); fprintf('\n');
            fprintf(fout, 'OVERALL');
            for x=1:length(metsBenchmark)
                fprintf(fout,'\t%f', metsBenchmark(x));
            end
            fprintf(fout, '\n\n');
        end

    end
end
fclose(fout);