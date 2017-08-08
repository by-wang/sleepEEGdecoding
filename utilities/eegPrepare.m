function [X,y] = eegPrepare(filedict,filename,opts)

% This function returns the EEG signals given the file path and parameters



load(filedict,filename);

% Nl = length(unwrap{1}.trial_idx);              % number of trials (left hand)
% Nr = length(unwrap{2}.trial_idx);              % number of trials (right hand)
% Nc = length(unwrap{1}.dims{2});                % number of channels  
% find the index of the strating and end point

tp = unwrap{1}.dims{4}/opts.unit;
[~, startIdx] = min(abs(tp-opts.offset));
[~, endIdx] = min(abs(tp-opts.offset-opts.width));


if isfield(opts,'trial')
    trial = opts.trial;
else
    trial = 'original';
end


% different types of signals
if strcmp(trial,'original')         
    Xl = squeeze(unwrap{1}.trial);
    Xr = squeeze(unwrap{2}.trial);
elseif strcmp(trial,'rms')
    
    Xl = squeeze(unwrap{1}.trialrms);
    Xr = squeeze(unwrap{2}.trialrms);
elseif strcmp(trial,'rmst')
    Xl = squeeze(unwrap{1}.trialrmst);
    Xr = squeeze(unwrap{2}.trialrmst);
end


if opts.bslc == 0                                                          % if we do baseline-correction

    Xl = Xl(:,:,startIdx:endIdx);
    Xr = Xr(:,:,startIdx:endIdx);
else
    bltimes = find(unwrap{1}.dims{4}<0);    
    BLl = squeeze(squeeze(mean(Xl(:,:,bltimes),3)));
    BLr = squeeze(squeeze(mean(Xr(:,:,bltimes),3)));
    
    for i = 1:size(Xl,1)
        tmpX = squeeze(Xl(i,:,:));
        Xl(i,:,:) = tmpX - BLl(i,:)';
    end
    
    for i = 1:size(Xr,1)
        tmpX = squeeze(Xr(i,:,:));
        Xr(i,:,:) = tmpX - BLr(i,:)';
    end
    
    Xl = Xl(:,:,startIdx:endIdx);
    Xr = Xr(:,:,startIdx:endIdx);
end


% the left/right reference fields, where the 4-th column is the indicator 
% pre-nap performance, the 5-th column is the indicator of post-nap 
% performance
r1 = unwrap{1}.ref;
r2 = unwrap{2}.ref;

if strcmp(opts.cond,'11') 
    idx1 = find(r1(:,4) == 1 & r1(:,5) ==1);
    idx2 = find(r2(:,4) == 1 & r2(:,5) ==1);
elseif strcmp(opts.cond,'01') 
    idx1 = find(r1(:,4) == 0 & r1(:,5) ==1);
    idx2 = find(r2(:,4) == 0 & r2(:,5) ==1);
elseif strcmp(opts.cond,'10') 
    idx1 = find(r1(:,4) == 1 & r1(:,5) == 0);
    idx2 = find(r2(:,4) == 1 & r2(:,5) == 0);
elseif strcmp(opts.cond,'00') 
    idx1 = find(r1(:,4) == 0 & r1(:,5) == 0);
    idx2 = find(r2(:,4) == 0 & r2(:,5) == 0);
else
    idx1 = unwrap{1}.trial_idx;
    idx2 = unwrap{2}.trial_idx;
end

Xl = Xl(idx1,:,:);
Xr = Xr(idx2,:,:);

if ~isempty(opts.Num)
    Xl = Xl(1:opts.Num,:,:);
    Xr = Xr(1:opts.Num,:,:);
end


Nl = size(Xl,1);
Nr = size(Xr,1);
    
yl = -ones(Nl,1);
yr = ones(Nr,1);

X = cat(1,Xl,Xr);
X = permute(X,[1,3,2]);

if ~isempty(opts.idx)
    X = X(:,:,opts.idx);
end

X = underSampling(X,opts.rate,opts.method);


y = [yl;yr];
