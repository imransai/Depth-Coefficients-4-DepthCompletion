%Function for saving Motion-Corrected Velodyne Points Projected into Image
%Plane.

kitti_depthset = '../../Data/KITTI_Depth/';
kitti_rawsequences = '../../Data/KITTI_Raw/';
dataset_type = 'train';
%Manufacture Training Data Path


pathlist = struct('kitti_depthset',kitti_depthset,'kitti_rawsequences',kitti_rawsequences); 
%The subsample is set for 16R (4 x 0 Skips), dataparam.skip_layer_rows = 5; 
%for 32R (1 x 1 Skips), set the dataparam.skip_layer_rows = 2;             
%for 64R (0 x 0 Skips), set the dataparam.skip_layer_rows = 1;
dataparam.skip_layer_rows = 5;
dataparam.skip_layer_cols = 1;
dataparam.Uniformflag = false;
dataparam.nsamples = 500;

dataparam.mot_correctedflag = false;
%Since the KITTI raw data has different sizes for different driving dates,
%we chose to use a fixed size for training. For evaluation, we chose the
%same size.
dataparam.STD_IMG_WIDTH = 1242;
dataparam.STD_IMG_HEIGHT = 352;
dataparam.truncated_height = 1;
dataparam.orig_normaliz = true;

colorsaveflag = true;
gtsaveflag = true;
save_trainvaldata_pts_imgplane(pathlist,dataparam, dataset_type, colorsaveflag,gtsaveflag);

function save_trainvaldata_pts_imgplane(pathlist, dataparam, dataset_type,colorsaveflag,gtsaveflag)                    

if nargin < 5 gtsaveflag = false; end
if nargin < 4 colorsaveflag = false; end 



if dataparam.Uniformflag
    subsample_folder = sprintf('Uniform_d%d',dataparam.nsamples);
else
    subsample_folder = sprintf('%dx%d_nSKips',dataparam.skip_layer_rows-1,dataparam.skip_layer_cols - 1);
     
    
end

gt_dir = 'groundtruth';

if dataparam.orig_normaliz   
   normalizefac = 256;
end

cam_catgry = {'image_02','image_03'};
folder2save = fullfile(pathlist.kitti_depthset,dataset_type);


data_dir = fullfile(pathlist.kitti_depthset, dataset_type);
drive_sequences = dir(data_dir);
drive_sequences = drive_sequences(3:end);
drive_sequences = {drive_sequences.name};

    
    for drive_indx = 1:length(drive_sequences)
            
            data_fulldir = fullfile(data_dir,drive_sequences{drive_indx},'proj_depth','groundtruth');
            drive_date = strsplit(drive_sequences{drive_indx},'_');
            drive_date = strjoin(drive_date(1:3),'_');
            kitti_rawdir = fullfile(pathlist.kitti_rawsequences, drive_date);
            
            for camid = 1:length(cam_catgry)
               data_camdir = fullfile(data_fulldir, cam_catgry{camid});
               
               savefolder_color = fullfile(folder2save,drive_sequences{drive_indx}, 'color',cam_catgry{camid});
               savefolder_depth = fullfile(folder2save, drive_sequences{drive_indx}, 'proj_depth',subsample_folder,cam_catgry{camid});
               savefolder_gtdepth = fullfile(folder2save, drive_sequences{drive_indx}, 'proj_depth', gt_dir, cam_catgry{camid});
               
               if (~exist(savefolder_color))
                  mkdir(savefolder_color); 
                   
               end
               
               if (~exist(savefolder_depth))
                  mkdir(savefolder_depth); 
                   
               end
               
               if (~exist(savefolder_gtdepth))
                   mkdir(savefolder_gtdepth);
                   
               end               
               
               
               filenames = dir(data_camdir)';
               filenames = filenames(3:end);
               filenames = {filenames.name};
               
               for file_id = 1:length(filenames)
                   
                   
                    
                    color_image = imread(fullfile(kitti_rawdir,drive_sequences{drive_indx},...
                    cam_catgry{camid},'data',filenames{file_id}));                   
                
                    annotated_dir = fullfile(pathlist.kitti_depthset,dataset_type,drive_sequences{drive_indx},...
                                        'proj_depth','groundtruth',cam_catgry{camid});
                    orig_depthmap = read_annnotatedgt(annotated_dir,filenames{file_id});
                    camnum = strsplit(cam_catgry{camid},'_');
                    camnum = str2double(camnum{2});
                    
                    fileindx = strsplit(filenames{file_id},'.');
                    fileindx = fileindx{1};
                    if ~ dataparam.Uniformflag
                        
                         velo_filename = fullfile(kitti_rawdir, drive_sequences{drive_indx},'velodyne_points/data',...
                                            [fileindx '.bin']);
                         velopts = read_velofiles(velo_filename);
                          
                          depth_zbuffered = subsample_4m_lidar(velopts, dataparam, fullfile(kitti_rawdir,drive_sequences{drive_indx}), ...
                                            kitti_rawdir, camnum, color_image, fileindx);
                    else
                
                        depth_zbuffered = bernoulli_picks(orig_depthmap, dataparam.nsamples);
                
                
                    end
                    
                    
                    [h, w] = size(orig_depthmap);                                       
                    
                    fileindx = strsplit(filenames{file_id},'.');
                    fileindx = fileindx{1};
                     
                    depth_zbuffered = padarray(depth_zbuffered,[0, ...
                        dataparam.STD_IMG_WIDTH - w],0,'pre');
                    
                    if h - dataparam.STD_IMG_HEIGHT + 1 > 0
                       
                        depth_zbuffered = depth_zbuffered((h - dataparam.STD_IMG_HEIGHT + 1):end,:);
                        
                    end
                    
                    
                    depth_zbuffered = full(depth_zbuffered)*normalizefac;
                    
                    if colorsaveflag
                        color_image = padarray(color_image,[0, ...
                                               dataparam.STD_IMG_WIDTH - w, 0],0,'pre');
                        if h - dataparam.STD_IMG_HEIGHT + 1 > 0
                            color_image = color_image((h - dataparam.STD_IMG_HEIGHT + 1):end,:,:);
                        end
                        imwrite(color_image,[savefolder_color '/' filenames{file_id}]);
                    end
                    
                    if gtsaveflag
                        
                        orig_depthmap = padarray(orig_depthmap,[0, dataparam.STD_IMG_WIDTH - w],0,'pre');
                        if h - dataparam.STD_IMG_HEIGHT + 1 > 0
                            orig_depthmap = orig_depthmap((h - dataparam.STD_IMG_HEIGHT + 1):end,:);
                        end
                        imwrite(uint16(orig_depthmap*normalizefac),[savefolder_gtdepth '/' filenames{file_id}]);
                        
                    end
                    

                    
                    imwrite(uint16(depth_zbuffered),[savefolder_depth '/' filenames{file_id}]);
                    
                    fprintf('Finished Writing imgseq %d of drive %s of cam %s\n',str2double(fileindx),...
                    drive_sequences{drive_indx}, cam_catgry{camid});
               end
            end
            
            

    end


end



function depth_zbuffered = subsample_4m_lidar(velopts, dataparam, seqdir, calib_dir, cam_id, color_image, fileno)
    
    
    [subsampled_pts,~] = lidar_subsample(velopts,dataparam.skip_layer_rows,dataparam.skip_layer_cols);
    
    
    subsampled_pts = subsampled_pts(:,1:3);
        
    
    
    %Extract the calibration parameters. The calibration params are
    %decoupled.
    calib_param = calib_read(calib_dir,cam_id);
    [pt_imgplane, depth] = project_2imgplane(subsampled_pts,calib_param,color_image);
    %Z_buffering
    depth_zbuffered = accumarray(round([pt_imgplane(:,2) pt_imgplane(:,1)]),depth,...
        [size(color_image,1) size(color_image,2)],@min,[],true);



end

function gt_depth = read_annnotatedgt(filepath,file_sequence)

gt_depth = double(imread(fullfile(filepath,file_sequence)));

gt_depth = gt_depth/256;

end

function calib_decoupledparam = calib_read(calib_dir,cam)  
 
%Load the calibration matrices
%load calibration
 calib = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
 Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));


% compute projection matrix velodyne->image plane
  R_cam_to_rect = eye(4);
  R_cam_to_rect(1:3,1:3) = calib.R_rect{1};  
  P_velo_to_img = calib.P_rect{cam+1}*R_cam_to_rect*Tr_velo_to_cam;
  intrinsics = [calib.P_rect{cam+1}(1) calib.P_rect{cam+1}(5) calib.P_rect{cam+1}(7) calib.P_rect{cam+1}(8)];
  t_rect = [calib.P_rect{cam+1}(10) calib.P_rect{cam+1}(11) calib.P_rect{cam+1}(12)];
  
  %Create a Structure for storing the decoupled calibration parameters
  calib_decoupledparam.R_cam_to_rect = R_cam_to_rect; %Stores the rotation matrix of the rectified camera 0 (velodyne calibrated wrt to cam 0)
  calib_decoupledparam.P_velo_to_img = P_velo_to_img; %Stores the rectified intrinsic matrix of the specified cam in coupled form
  calib_decoupledparam.intrinsics = intrinsics; %Stores the decoupled rectified intrinsics of the specified camera 
  calib_decoupledparam.t_rect = t_rect; %Stores the xlation shift between the specified camera and cam0
  calib_decoupledparam.Tr_velo_to_cam = Tr_velo_to_cam; %Stores the extrinsics of the parameters from velodyne to cam0.
  calib_decoupledparam.P_rect = calib.P_rect{cam+1};
  %intrinsics = [fx fy u0 v0];
  
end

function velopts = read_velofiles(velo_filename)
    
    fid = fopen(velo_filename,'rb');
    velopts = fread(fid,[4 inf],'single')';
    fclose(fid);
    


end


function depth_sampbern = bernoulli_picks(orig_depthmap,nsamples)

    depth_sampbern = zeros(size(orig_depthmap));
    valid_depth = orig_depthmap(orig_depthmap ~= 0);
    valid_depthinds = find(orig_depthmap ~= 0);

    prob = nsamples/length(valid_depth);
    y = binornd(ones(1,length(valid_depth)), prob);

    depth_picks = valid_depth(y >= prob);


    pick_inds = valid_depthinds(y >= prob);

    depth_sampbern(pick_inds) = depth_picks;

end

function [subsampled_pts,layer_id] = lidar_subsample(velopts,skip_row,skip_col)
 if nargin < 3 skip_col = 1; end
 if nargin < 2 skip_row = 1; end
    
%     %First Sort the lidar points according to the grid
%     [rho, theta,z] = cart2pol(velopts(:,1),velopts(:,2),velopts(:,3));
    %Calculating azimuth angle of lidar
    az = atan2( velopts(:,2)', velopts(:,1)' );
    
    %Storing the azimuth angle.        
    newRowInds = [1 find(az(2:end) >=0 & az(1:end-1) < 0) + 1];
    endRowInds = [newRowInds(2:end)-1 numel(az)];
    
    subsampled_pts = [];
    layer_id = zeros(round(length(newRowInds)/skip_row),1);    
    count = 1;
    for ii = 1:skip_row:length(endRowInds)
       extracted_pts = velopts(newRowInds(ii):skip_col:endRowInds(ii),:);
       dontkeeppts = extracted_pts(:,1)<5;
       extracted_pts(dontkeeppts,:) = [];
       subsampled_pts = [subsampled_pts; extracted_pts];
       
       layer_id(count) = length(extracted_pts);
       count = count + 1;
        
    end
    layer_id(layer_id == 0) = [];


end

function [imgplane_uvcoord, z] = project_2imgplane(p_in,calib_param,color_img)
    
       
    TransMatrix = calib_param.P_velo_to_img;
    % dimension of data and projection matrix
    dim_norm = size(TransMatrix,1);
    dim_proj = size(TransMatrix,2);

    % do transformation in homogenuous coordinates
    p2_in = p_in;
    
    if size(p2_in,2)<dim_proj
        p2_in(:,dim_proj) = 1;
    end
    
    velo_camptcloud = (TransMatrix*p2_in')';   
       
    
    % normalize homogeneous coordinates:
    imgplane_uvcoord = round(velo_camptcloud(:,1:dim_norm-1)./(velo_camptcloud(:,dim_norm)*ones(1,dim_norm-1)));
    
    z = velo_camptcloud(:,3);
    %Prune out the lidar points outside the image plane
    keep = imgplane_uvcoord(:,2)>0 & imgplane_uvcoord(:,2) <= size(color_img, 1) & ...
           imgplane_uvcoord(:,1)>0 & imgplane_uvcoord(:,1) <= size(color_img, 2);
    
       
    imgplane_uvcoord = imgplane_uvcoord(keep,:);
    z = z(keep);
    
    

end

function ptcloud_cam = project_2camcoord(ptcloud_imgplane,calib_param)
    if nargin < 3 layer_info = []; end
    %Decouple parameters from camera calibration.    
    intrinsics = calib_param.intrinsics; t_rect = calib_param.t_rect;
    R_cam_to_rect = calib_param.R_cam_to_rect;
    %Project To Lidar Plane for Subsampling
    yc = ptcloud_imgplane(:,3)/intrinsics(2).*(ptcloud_imgplane(:,2) - intrinsics(4) - t_rect(2)./ptcloud_imgplane(:,3));
    xc = ptcloud_imgplane(:,3)/intrinsics(1).*(ptcloud_imgplane(:,1) - intrinsics(3) - t_rect(1)./ptcloud_imgplane(:,3));
    
    ptcloud_cam = [xc yc ptcloud_imgplane(:,3)];
     ptcloud_cam = R_cam_to_rect(1:3,1:3)'*ptcloud_cam';
     ptcloud_cam = ptcloud_cam';
    
end

function imgcoord_uv = project_2imgplane4m_camcoord(ptcloud_cam,calib_param)
    % dimension of data and projection matrix
    dim_norm = size(calib_param.P_rect,1);
    
    ptcloud_cam = [ptcloud_cam ones(length(ptcloud_cam),1)];
    imgcoord_uv = calib_param.P_rect*calib_param.R_cam_to_rect*ptcloud_cam';
    imgcoord_uv = imgcoord_uv';
    imgcoord_uv = imgcoord_uv(:,1:dim_norm-1)./(imgcoord_uv(:,dim_norm)*ones(1,dim_norm-1));

end

function ptcloud_correctedlayer = occlusion_correction(ptcloud_cam,split_bin)
    
    if nargin < 3 color_img = []; end
    if nargin < 2 || isempty(split_bin) split_bin = 10; end
    
    %First Calculate the elevation angle of all the points
    
    elev_angle = atan2(ptcloud_cam(:,2),ptcloud_cam(:,3));
    
    [~,edges] = histcounts(elev_angle,split_bin);
    indxlayer = 1:length(edges)-1;
    x = [1:length(elev_angle)]'/length(elev_angle);
    keeplayer = arrayfun(@(x) elev_angle >= edges(x) & elev_angle < edges(x+1),indxlayer,'UniformOutput',false);
    C = clusterdata([x elev_angle],'linkage','single','maxclust',10);
    finlabel = labelconversion(keeplayer,C);
    ptcloud_layer = arrayfun(@(x) ptcloud_cam(finlabel == x,:),indxlayer,'UniformOutput',false);
    az_layer = cellfun(@(x) atan2(x(:,1),x(:,3)),ptcloud_layer,'UniformOutput',false);
    
    keep = cellfun(@(x) occlude_eliminator(x),az_layer,'UniformOutput',false);
    
    ptcloud_correctedlayer = cellfun(@(x,y) ptcloud_layer{y}(x,:),keep,num2cell(indxlayer),'UniformOutput',false);
    ptcloud_correctedlayer = cat(1,ptcloud_correctedlayer{:});

end

function keep = occlude_eliminator(az_layer)
    keep = zeros(length(az_layer),1,'logical'); keep(1) = 1;
   
    azabs_layer = abs(az_layer);
    azabs_layer1 = azabs_layer(az_layer>=0 & az_layer<pi);
    azabs_layer1 = azabs_layer1(end:-1:1);
    keep1 = zeros(length(azabs_layer1),1,'logical'); keep1(1) = 1;
    azabs_layer2 = azabs_layer(az_layer>=-pi & az_layer<0);
    %azabs_layer2 = azabs_layer2(end:-1:1); 
    keep2 = zeros(length(azabs_layer2),1,'logical');
    
    pt_1 = azabs_layer1(1);pt_2 = azabs_layer1(2); keep1(1) = 1; keep1(end) = 1;
    for ii = 2:length(azabs_layer1) -1
        
        if (pt_2 - pt_1) > 0
           keep1(ii) = 1;
           pt_1 = azabs_layer1(ii);
           pt_2 = azabs_layer1(ii+1);
        else
           pt_2 = azabs_layer1(ii+1);
        end
        
    end
    keep1 = keep1(end:-1:1);
    pt_1 = azabs_layer2(1);pt_2 = azabs_layer2(1); keep2(1) = 1; keep2(end) = 1;
    for ii = 1:length(azabs_layer2) - 1
        if (pt_2 - pt_1) > 0
           keep2(ii) = 1;
           pt_1 = azabs_layer2(ii);
           pt_2 = azabs_layer2(ii+1);
        else
           pt_2 = azabs_layer2(ii+1);
        end
        
    end
    
    
    keep = [keep1;keep2];
    %{
    az_layer = az_layer(end:-1:1);
    pt_1 = az_layer(1); pt_2 = az_layer(2);
    for ii = 1:length(az_layer)-1
        
%         [~,indx] = min(az_layer(ii+1:end) - pt_1) ;
%         keep(ii + indx) = 1;
        %pt_1 = az_layer(ii+1);
        
        if ((pt_2 - pt_1) >= 0)
           keep(ii) = 1;
           pt_2 = az_layer(ii+1);
           pt_1 = az_layer(ii);
        else
           pt_2 = az_layer(ii+1); 
        end
        
        
    end
    keep = keep(end:-1:1);
    %}
end

function finlabels = labelconversion(keeplayer,C)

if nargin<2 | isempty(C) C = []; end

finlabels = zeros(length(keeplayer),1);

for ii = 1:length(keeplayer)
    finlabels(keeplayer{ii}) = ii;
end

if ~isempty(C)
    
   finlabels(finlabels == 10) = 0;
   finlabels(C == 10) = 10;
   finlabels(finlabels == 9) = 0;
   finlabels(C == 9) = 9;
   finlabels(finlabels == 8) = 0;
   finlabels(C == 8) = 8;
end


end

function plot_depthpoints_onColor(depth_img, zval)
if nargin< 2 
    zmin = 5; zmax = 70; 
else
    zmin = zval(1); zmax = zval(2);
end

[row, col] = find(depth_img);
ind = sub2ind(size(depth_img), row, col);
depth = depth_img(ind);

pt_imgGround = [col row depth];
c = max(0, min(zmax, pt_imgGround(:,3)));
plotPointsColor(round(pt_imgGround(:,1)), round(pt_imgGround(:,2)), [], c, {'.','markersize',8}, jet(256), [zmin zmax]);



end

function plot_points_in_imageplane(img_coord,errorval,max_val)
if nargin < 3 max_val = 100; end 
% plot points
    cols = jet;
    for i=1:size(errorval,1)
      col_idx = min([round(64*errorval(i)/max_val) 64]);
      col_idx = max([col_idx 1]);
      try
      plot(img_coord(i,1),img_coord(i,2),'o','LineWidth',4,'MarkerSize',5,'Color',cols(col_idx,:));
      catch
          here
      end
    end
end


function generate_dc2plotpoints(depthmap, dstep, nstep, cshow)
if nargin < 4 cshow = 20; end
if nargin < 3 nstep = 80; end
if nargin < 2 dstep = 100; end

dc = depth2Discrete(depthmap,dstep,nstep);
dc_vals = dc.getVals(1);

dc_vals = reshape(dc_vals,[],nstep + 2);

dc_pick = dc_vals(:,cshow);

ind = find(dc_pick);
[row, col] = ind2sub(size(depthmap),ind);

dc_pickvals = dc_pick(ind);
plot_points_in_imageplane([col, row],dc_pickvals,0.5);


end