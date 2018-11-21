% drawss bbox on an image (im1) and returns the image file
% bbox: a matrix of size n*5
% default line width (lw) is 2
%function im1 = show_bbox_on_image(im1, bbox, bws, col, lw)
function im1 = show_bbox_on_image(ids, im1, bbox, col, lw)

if ~exist('lw')
  lw = 2;
end

m1 = floor((lw-1)/2)+1;   %% reduce 1 for the pixel itself
m2 = ceil((lw-1)/2)+1;

[sz1,sz2,sz3] = size(im1);
sz = size(bbox, 1);

bbox = round(bbox);

for j = floor(size(bbox,2)/4):-1:1 %%for all parts
  for i = 1:sz
    x1 = bbox(i, (j-1)*4+1);
    y1 = bbox(i, (j-1)*4+2);
    x2 = bbox(i, (j-1)*4+3);
    y2 = bbox(i, (j-1)*4+4);
%     disp(bbox(i,end-1));
%     input('Input?');
    for k = 1:3  %% RGB channels
%       im1(max(1,y1-m1):min(sz1,y1+m2),  max(1,x1):min(sz2,x2),        k) = col(k, bbox(i,end-1));
%       im1(max(1,y2-m1):min(sz1,y2+m2),  max(1,x1):min(sz2,x2),        k) = col(k, bbox(i,end-1));
%       im1(max(1,y1):min(sz1,y2),        max(1,x1-m1):min(sz2,x1+m2),  k) = col(k, bbox(i,end-1));
%       im1(max(1,y1):min(sz1,y2),        max(1,x2-m1):min(sz2,x2+m2),  k) = col(k, bbox(i,end-1));
      
      im1(max(1,y1-m1):min(sz1,y1+m2),  max(1,x1):min(sz2,x2),        k) = col(k, bbox(i,end));
      im1(max(1,y2-m1):min(sz1,y2+m2),  max(1,x1):min(sz2,x2),        k) = col(k, bbox(i,end));
      im1(max(1,y1):min(sz1,y2),        max(1,x1-m1):min(sz2,x1+m2),  k) = col(k, bbox(i,end));
      im1(max(1,y1):min(sz1,y2),        max(1,x2-m1):min(sz2,x2+m2),  k) = col(k, bbox(i,end));
    end
  end  
end

for count = 1:sz
    position = [bbox(count,1) bbox(count,2)];
    num = bbox(count,5);
    if num == ids{1} || num == ids{2}
        im1 = insertText(im1,position,num2str(num),'FontSize',15,'BoxColor','blue','TextColor','green');
    end

%     im1 = insertText(im1,position,num2str(num),'FontSize',10,'BoxColor','blue','TextColor','white');
end

end


