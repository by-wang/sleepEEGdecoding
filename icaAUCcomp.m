function icaAUCcomp(job_id)

% This function computes the AUC values of each ICA component under
% different pre/post-nap performances


%--------

% pre/post-nap performances. The first digit encodes the pre-nap
% performance, the second digit encodes the post-nap performance. 
% 1: remember, 0: forget
conds = {'10'};


% add auxiliary functions for EEG processing and analysis
addpath(genpath('/mnt/bucket/people/boyuw/code'));





% load ICAed EEG signals
filename = 'Boyu-1-0-0-0-3-1';
load(['/mnt/bucket/labs/norman/jantony/motorExp/analysis/',filename])



% read the task ID for parallel computing
SGE_TASK_ID = str2double(getenv('SLURM_ARRAY_TASK_ID'));%SLURM
so = SGE_TASK_ID;    



X = squeeze(data_a_a(:,so,:));              % ICAed EEG signals
Y = inds_a;                                 % class labels
Y(Y==2) = -1;

clear data_a_a;


subID =  unique(rows_a);                    % subject IDs
width = [1,25,50,125,250,500];              % different window lengths 



numSub = length(subID);                     % number of subjects
numT = size(X,2);                           % number of time points
numW = length(width);                       % number of window lengths
numConds = length(conds);                   % number of conditions


AUC = zeros(numSub,numW,numT);
empty_id = zeros(numSub,2);


for i = 1:numSub
    % extract the EEG signals of the i-th subject
    Idx = subID(i);
    Xi = X(rows_a == Idx,:);
    yi = Y(rows_a == Idx);
    ref = refs_a{i}.refs;
    
    
    % extract the EEG signals under different conditions
    tmpX = [];
    ytmp = [];
    for c = 1:numConds
        [Xc, yc] = ICAedEEGPrepare(Xi,yi,ref,conds{c});
        tmpX = cat(1,tmpX,Xc);
        ytmp = cat(1,ytmp,yc);
    end
    
    % compute the AUC values at different time points, with different window lengths
    if length(unique(ytmp))>1
        for j = 1:numW
            wid = width(j);
            for k = 1:numT
                endIdx = min(k+wid-1,numT);
                tmpx = tmpX(:,k:endIdx);
                tmpx = mean(tmpx,2);    % average the signals over time

                [~,~,~,A] = perfcurve(ytmp,tmpx,1);
                AUC(i,j,k) = A;
            end
        end
    else
        empty_id(i,1) = 1;
        empty_id(i,2) = Idx;
    end
end

clear X
clear tmpX
clear Xc
clear Xi
clear tmpx
clear ica_a

condname = [];
for c = 1:numConds
    condname = [condname,conds{c}];
end
% save the results
resultpath = '/mnt/bucket/labs/norman/boyuw/results/Results2017_06_05/LR/partial_ICA_AUCcheck/';
mkdir(resultpath);
save([resultpath,'/LR_',filename,'_',condname,'_',num2str(job_id),'_',num2str(so)]);

