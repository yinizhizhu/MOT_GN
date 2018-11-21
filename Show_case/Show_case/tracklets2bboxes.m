function bboxes_tracked = tracklets2bboxes(tracklets,fnum,tracking_id)

count = 1;

for frame = 1:fnum
    for i = 1:tracking_id
        for j = 1:size(tracklets(i).tracklet,1)
            if(tracklets(i).tracklet(j,6) == frame)
                bboxes_tracked(frame).bbox(count,1) = tracklets(i).tracklet(j,1);
                bboxes_tracked(frame).bbox(count,2) = tracklets(i).tracklet(j,2);
                bboxes_tracked(frame).bbox(count,3) = tracklets(i).tracklet(j,3);
                bboxes_tracked(frame).bbox(count,4) = tracklets(i).tracklet(j,4);
                bboxes_tracked(frame).bbox(count,5) = i; % belongs to which tracklets
                
%                     bboxes_tracked(frame).bbox(count,6) = tracklets(i).tracklet(j,7); % test
                count = count+1;
                break;
            end
        end
    end
    count = 1;
end

end