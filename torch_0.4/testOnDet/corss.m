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
datasets = {'training', 'validation'};

if ~exist(metrics_dir, 'dir')
    mkdir(metrics_dir);
    fprintf('There is no %s! And we create this dir\n',metrics_dir);
end

seqs = [2, 4, 5, 9, 10, 11, 13]; % the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]; % the length of the sequence
types = {'dets', 'gts'};


for i=1:length(seqs)    % Iteration for sequence
    % the current training dataset
    seq_index = sprintf('%02d', seqs(i));
    len = lengths(i);
%     disp(seq_index);
    
    % Output the result of motmetrics into the text file
    motmetrics_dir = fullfile(metrics_dir, sprintf('%s.txt', seq_index));
    fout = fopen(motmetrics_dir, 'w');

    % the length of the training dataset
    tts = [];
    tts(1) = len; % for the cross evaluation

    f_dir = fullfile(basic_dir, seq_index);
    disp(f_dir);
    for j=1:length(tts) %Iteration for training sequence
        tag = 1;
        tt = tts(j);
        
        if tt*2 > len
            if tt == len
                tag = 0;
            else
                continue;
            end
        end
        fprintf('The sequence: %s - The length of the training data: %d\n', seq_index, tt);

        s_dir = fullfile(f_dir, sprintf('%d', tt));
%         disp(s_dir);


        for data_select=1:2
            dataset = datasets{data_select};

            for k=1:length(types)   %Iteration for different condition
                % Read sequence list
                gtMat = [];
                resMat = [];

                % Evaluate sequences individually
                allMets = [];
                metsBenchmark = [];
                metsMultiCam = [];

                seq_name = strcat(seq_index, sprintf('_%d_%s', tt, dataset));
                fprintf(fout, 'Sequence');
                for x=1:length(metrics)
                    fprintf(fout, '\t%s', metrics{x});
                end
                fprintf(fout, '\n');
                
                type = sprintf('motmetrics_cross_%s', types{k});
                t_dir = fullfile(s_dir, type);
    %             disp(t_dir);
                
                ind = 0;
                for x=1:7
                    if x == i
                        continue;
                    end
                    ind = ind+1;
                    cross_index = sprintf('%02d', seqs(x));
                    gt_training = fullfile(t_dir, sprintf('gt_%s.txt', cross_index));
                    res_training = fullfile(t_dir, sprintf('res_%s.txt', cross_index));
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
                    allMets(ind).name = seq_name;
                    allMets(ind).m    = mets;
                    allMets(ind).IDmeasures = metsID;
                    allMets(ind).additionalInfo = additionalInfo;
                    fprintf('%s_%s\n', seq_name, types{k}); printMetrics(mets);
                    fprintf(fout, '%s_%s', seq_index, cross_index);
                    for l=1:length(mets)
                        fprintf(fout,'\t%f', mets(l));
                    end
                    fprintf(fout, '\n');
                end
                % Overall scores
                metsBenchmark = evaluateBenchmark(allMets, world);
                fprintf(' ********************* Your %s Results *********************\n', type);
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
end