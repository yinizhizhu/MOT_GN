%%% This function is used to deal with those tracks which miss some
%%% detections in some frames. THe function can automaticlly predict the
%%% miss detection's position and write them in the tracklets.
%%% ATTENTION : The prediction's detection number is supposed to be -inf.
%%% This function is written by Gao Xu, 2016-04-17.

function tracklets = fill_vacancy_in_one_track(tracklets, tracking_id)

for i = 1:tracking_id
    buff = []; % the first num in each vacancy area.
    count = []; % how many number need to be add in each vacancy area.
    beforeRowBuff = [];
    afterRowBuff = [];
    ptr = 1; % vacancy area numbers.
    counter = 0;
    
    if ~isempty(tracklets(i).tracklet)
        
        for j = min(tracklets(i).tracklet(:,6)):max(tracklets(i).tracklet(:,6))
            if isempty(find(tracklets(i).tracklet(:,6) == j)) && ~isempty(find(tracklets(i).tracklet(:,6) == j-1))
                buff(ptr) = j;
                [beforeRowBuff(ptr) y] = find(tracklets(i).tracklet(:,6) == j-1);
            end
            if isempty(find(tracklets(i).tracklet(:,6) == j))
                counter = counter+1;
            end
            
            if isempty(find(tracklets(i).tracklet(:,6) == j)) && ~isempty(find(tracklets(i).tracklet(:,6) == j+1))
                count(ptr) = counter;
                [afterRowBuff(ptr) y] = find(tracklets(i).tracklet(:,6) == j+1);
                ptr = ptr+1;
                counter = 0;
            end
        end
        
        % fill the vacancy
        if ~isempty(buff)
            for k = 1:length(buff) % for each detections which should have been detected in the tracklets but not.
                
                a = beforeRowBuff(k);
                b = afterRowBuff(k);
                deltaX1 = (tracklets(i).tracklet(b,1) - tracklets(i).tracklet(a,1)) / (count(k) + 1);
                deltaY1 = (tracklets(i).tracklet(b,2) - tracklets(i).tracklet(a,2)) / (count(k) + 1);
                deltaX2 = (tracklets(i).tracklet(b,3) - tracklets(i).tracklet(a,3)) / (count(k) + 1);
                deltaY2 = (tracklets(i).tracklet(b,4) - tracklets(i).tracklet(a,4)) / (count(k) + 1);
                
                for p = 1:count(k)
                    
                    currentRow = size(tracklets(i).tracklet,1) + 1;
                    tracklets(i).tracklet(currentRow,1) = tracklets(i).tracklet(a,1) + deltaX1 * p;
                    tracklets(i).tracklet(currentRow,2) = tracklets(i).tracklet(a,2) + deltaY1 * p;
                    tracklets(i).tracklet(currentRow,3) = tracklets(i).tracklet(a,3) + deltaX2 * p;
                    tracklets(i).tracklet(currentRow,4) = tracklets(i).tracklet(a,4) + deltaY2 * p;
                    tracklets(i).tracklet(currentRow,5) = tracklets(i).tracklet(a,5);
                    tracklets(i).tracklet(currentRow,6) = buff(k) + p - 1;
                    tracklets(i).tracklet(currentRow,7) = -inf;
                    
                end
            end
        end
    end
end

end