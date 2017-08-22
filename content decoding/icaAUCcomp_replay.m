function icaAUCcomp_replay(job_id)

% This function computes the AUC values of each ICA component for content 
% decoding (left/right) based on the output of relay classifier. It uses 
% the output of  replay classifier to select the trials that most likely 
% have replay during the sleep.


%--------

% add auxiliary functions for EEG processing and analysis
addpath(genpath('/mnt/bucket/people/boyuw/code/utilities'));


% load the performance (AUC values) of replay classifier 
processed_replaypath = '/mnt/bucket/labs/norman/boyuw/results/Results2017_05_15/Motor/L1/withinSub/wobslc/Processed_Voltage_none/11vs10/Bootstrapped_Results.mat';
load(processed_replaypath,'total_AUC1');            




% the path of the output of the replay classifier
replaypath = '/mnt/bucket/labs/norman/boyuw/results/Results2017_05_15/Motor/L1/withinSub/wobslc/Voltage_none/11vs10';


list = dir(replaypath);
if strcmp(list(3).name,'.DS_Store')
    list(1:3) = [];
else
    list(1:2) = [];
end


conds = {'11','10'};



levels = 0.1:0.1:1;                                         % ratios of EEG signals used for computing AUC

replayAUC = squeeze(mean(total_AUC1,1));                    % the AUC values of 11/10 classifier
replayAUC = squeeze(replayAUC(:,:,4));                      % the AUCs of logistic regression without regularization


% load ICAed EEG signals
filename = 'Boyu-1-0-0-0-3-1';
load(['/mnt/bucket/labs/norman/jantony/motorExp/analysis/',filename]);


% read the task ID for parallel computing, so is the ICA component index
SGE_TASK_ID = str2double(getenv('SLURM_ARRAY_TASK_ID'));%SLURM
so = SGE_TASK_ID;   

X = squeeze(data_a_a(:,so,:));                              % ICAed EEG signals
Y = inds_a;                                                 % class labels - left/right
Y(Y==2) = -1;

clear data_a_a;



subID =  unique(rows_a);                                    % subject IDs
subID = sort(subID,'ascend');
width = [1,25,50,125,250];                                  % different window lengths 


numSub = length(subID);                                     % number of subjects
numT = size(X,2);                                           % number of time points
numW = length(width);                                       % number of window lengths
numConds = length(conds);                                   % number of conditions
numL = length(levels);                                      % number of ratios

AUC = zeros(numSub,numW,numT,numL);
NumPos = zeros(numSub,numW,numL);
NumNeg = zeros(numSub,numW,numL);


empty_id = zeros(numL,numSub,2);                                           % 1 = no output from replay classifier, 2 = labels do not match, 3 = only one class (left or right)
match_indicator = zeros(numSub,1);


for nf = 1:numSub
    % extract the ICAed EEG signals of the nf-th subject
    Idx = subID(nf);
    Xi = X(rows_a == Idx,:);                                               
    yi = Y(rows_a == Idx);
    ref = refs_a{nf}.refs;
    
    
    [Y_hat,y_true] = loadReplay(replaypath,list,Idx);                      % load the output of the replay classifier for the nf-th subject                  
    
    
    if isempty(y_true)                                                     % check if the subject is found in the replay results
        empty_id(:,nf,1) = 1;
        empty_id(:,nf,2) = Idx;
        AUC(nf,:,:,:) = nan;
    else
        
        % extract the EEG signals under different conditions
        Xica = [];
        yica = [];
        numc = zeros(1,numConds);
        for c = 1:numConds
            [Xc, yc] = ICAedEEGPrepare(Xi,yi,ref,conds{c});                % extract the ICAed L/R EEG signals with specific conditions (e.g., 11, 10)

            Xica = cat(1,Xica,Xc);
            yica = cat(1,yica,yc);
            numc(c) = length(yc);
        end
        
        % double-check if labels from two sources match
        zi = [-ones(numc(1),1);ones(numc(2),1)];                           
        if sum(abs(zi-y_true)) ~= 0
            match_indicator(nf) = 1;                                       
            empty_id(:,nf,1) = 2;
            empty_id(:,nf,2) = Idx;
            AUC(nf,:,:,:) = nan;
            continue
        end
        
        
        numTrial = length(yica);
        if length(unique(yica))>1                                          % check if the data set contain the EEG signals from both classes (i.e., left/right)
            
            % compute the AUC values at different time points, with
            % different window lengths and ratios
            
            for j = 1:numW
                % Use the output and AUC of replay classifier with window
                % length = 1000ms
                tmpAUC = replayAUC(:,numW);                                
                [~, timeIdx] = max(tmpAUC);
                y_replay = Y_hat{timeIdx,numW,4};
                
                % Sort the output of replay classifier in ascend order,
                % since the negative (majority) class indicates trial with
                % replay, while the positive (minority) class indicates the
                % trial without replay. In other words, the lower the value
                % of the replay classifier's output, the more likely the
                % trial contains replay.
                
                [~, s_idx] = sort(y_replay,'ascend');

                wid = width(j);
                for nl = 1:numL
                    ratio = levels(nl);
                    ntmp = round(numTrial*ratio);
                    
                    
                    Yr = yica(s_idx(1:ntmp));
                    Xr = Xica(s_idx(1:ntmp),:);
                    
                    
                    NumPos(nf,j,nl) = sum(Yr==1);
                    NumNeg(nf,j,nl) = sum(Yr==-1);
                    
                    if sum(Yr==-1) == 0 || sum(Yr==1) == 0
                        
                        AUC(nf,j,:,nl) = nan;
                    else
                    
                        for k = 1:numT
                            endIdx = min(k+wid-1,numT);

                            tmpx = Xr(:,k:endIdx);
                            tmpx = mean(tmpx,2);                           % average the signals over time

                            [~,~,~,A] = perfcurve(Yr,tmpx,1);
                            AUC(nf,j,k,nl) = A;
                        end
                    end
                end
            end
        else
            empty_id(:,nf,1) = 3;
            empty_id(:,nf,2) = Idx;
            AUC(nf,:,:,:) = nan;
        end
    end
end

clear X
clear Xi
clear ica_a

condname = [];
for c = 1:numConds
    condname = [condname,conds{c}];
end
            

resultpath = '/mnt/bucket/labs/norman/boyuw/results/Results2017_06_05/ReplyLR/partial_ICA_AUCcheck';



% save the results
mkdir(resultpath);
save([resultpath,'/LR_',filename,'_',condname,'_',num2str(job_id),'_',num2str(so)]);

    
            



