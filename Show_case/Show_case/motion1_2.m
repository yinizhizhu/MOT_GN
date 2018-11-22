%% Setting and gloabal parameters
clc
clear
clear all;
% 

year = 17;
% types = {'SDP', 'FRCNN', 'DPM'};
% datasets = {'01', '03', '06', '07', '08', '12', '14'};
% frames = {450, 1500, 1194, 500, 625, 900, 750};

trackdir = sprintf('/home/lee/Desktop/Test17/Case/%s/', 'Motion1_2');
vid_format = 'jpg';
fill_vac = 0; % =1: fill the vacancy in one track; =0: do not do that!

ids = {{20, 23}, {13, 13}};

%% for each dataset, use tracking result
for dbCount = 1:2

    display('----------------------------------------------------');
    result_path = [trackdir sprintf('%d.txt', dbCount)];
    display(result_path);
    result = load(result_path);
    out_dir = [trackdir sprintf('%d/', dbCount)];

    %% If you want to show the detections, open this part
% {
%     b = 0;
%     for i = 1:size(result,1)
%         result(i,2)=i - b;
%         if mod(i,1000) == 0
%             b = b + 1000;
%         end
%     end
% }   
    %% Construct tracklets
    tracklets = [];

    for i = 1:max(result(:,2))
        inds = find(result(:,2) == i);
        for j = 1:length(inds)
            tracklets(i).tracklet(j,1) = result(inds(j),3); % left-top point col
            tracklets(i).tracklet(j,2) = result(inds(j),4); % left-top point row
            tracklets(i).tracklet(j,3) = result(inds(j),3)+result(inds(j),5); % right-bottom point col
            tracklets(i).tracklet(j,4) = result(inds(j),4)+result(inds(j),6); % right-bottom point row
            tracklets(i).tracklet(j,5) = 50; % tracklet rank, set to be 50, useless now.
            tracklets(i).tracklet(j,6) = result(inds(j),1); % frame 

%                 tracklets(i).tracklet(j,7) = result(inds(j), 11); % test
        end
    end

    tracking_id = max(result(:,2));
    fnum = max(result(:,1));
    display(tracking_id);
    display(fnum);

    %% Fill the vacancy in one track
    if fill_vac
        display('Filling the vacancy in one track ...');
        tracklets = fill_vacancy_in_one_track(tracklets, tracking_id);
    end

    bboxes_tracked = [];

    input_frames = [trackdir '%0.6d.' vid_format];
    output_vidname  = [out_dir 'tracked.avi'];

    bboxes_tracked = tracklets2bboxes(tracklets,fnum,tracking_id);
    
    display(tracking_id);
    show_bboxes_on_video(input_frames, bboxes_tracked, tracking_id, output_vidname, 25, -inf, out_dir, ids{dbCount});
    

%     unix(['rm -r ' output_path]);
end