function icaAUCcomp_LOSO(job_id)

% This function computes the AUC values of each ICA component for content 
% decoding (left/right) using leave-one-subject-out (LOSO) classifier. Each
% time it leaves one subject as the testing data, using other subjects as
% training data to train a logistic regression classifier.



%--------

% add auxiliary functions for EEG processing and analysis
addpath(genpath('/mnt/bucket/people/boyuw/code'));


% For comparison purpose, we only consider the condition 11 & 10. where the 
% first digit encodes the pre-nap performance, the second digit encodes the 
% post-nap performance. 
% 1: remember, 0: forget
conds = {'11','10'};



% load ICAed EEG signals
filename = 'Boyu-1-0-0-0-3-1';
load(['/mnt/bucket/labs/norman/jantony/motorExp/analysis/',filename])


% Initialize the parameters for logistic regression classifier
opts=[];

% Starting point
opts.init=2;        % starting from a zero point
% Termination 
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=100;    % maximum number of iterations
% Normalization
opts.nFlag=0;       % without normalization
% Regularization
opts.rFlag=0;       % the input parameter 'rho' is a ratio in (0, 1)
opts.mFlag=1;       % smooth reformulation 
opts.lFlag=1;       % adaptive line search
opts.tFlag=2; 




% read the task ID for parallel computing, so is the ICA component index
SGE_TASK_ID = str2double(getenv('SLURM_ARRAY_TASK_ID'));%SLURM
so = SGE_TASK_ID;    



X = squeeze(data_a_a(:,so,:));          % ICAed EEG signals
Y = inds_a;                             % class labels - left/right
Y(Y==2) = -1;

clear data_a_a;

subID =  unique(rows_a);                % subject IDs
width = [1,25,250,500];                 % different window lengths 



numSub = length(subID);                     % number of subjects
numT = size(X,2);                           % number of time points
numW = length(width);                       % number of window lengths
numConds = length(conds);                   % number of conditions



total_acc = zeros(numSub,numW,numT);
Y_hat = cell(numSub,numW,numT);
Y_true = cell(numSub,numW,numT);

A1 = zeros(numSub,numW,numT);
A2 = zeros(numSub,numW,numT);



Xi = cell(1,numSub);
Yi = cell(1,numSub);


for i = 1:numSub
    
    % extract the ICAed EEG signals of the i-th subject
    Idx = subID(i);
    xi = X(rows_a == Idx,:);
    yi = Y(rows_a == Idx);
    ref = refs_a{i}.refs;
    
    
    % extract the EEG signals under different conditions
    Xica = [];
    yica = [];
    for c = 1:numConds
        [Xc, yc] = ICAedEEGPrepare(xi,yi,ref,conds{c});
        Xica = cat(1,Xica,Xc);
        yica = cat(1,yica,yc);
    end
    
    Xi{i} = Xica;
    Yi{i} = yica;
end

clear Xica
clear yica



for i = 1:numSub
    
    
    % Prepare the training and testing data
    Xtest = Xi{i};
    ytest = Yi{i};
    
    Xtrain = [];
    ytrain = [];
    
    for ii = 1:numSub
        if ii ~= i
            Xtrain = cat(1,Xtrain,Xi{ii});
            ytrain = cat(1,ytrain,Yi{ii});
        end
    end
    
    
    % compute the AUC values at different time points, with different window lengths
    for j = 1:numW
        wid = width(j);
        for k = 1:numT
            endIdx = min(k+wid-1,numT);
            
            Xtr = Xtrain(:,k:endIdx);
            Xtr = mean(Xtr,2);
            
            Xte = Xtest(:,k:endIdx);
            Xte = mean(Xte,2);
            [w_train, c_train, ~, ~]= LogisticR(Xtr, ytrain, 0.000000001, opts);    % training

            h = sign(Xte*w_train+c_train);                                          % testing
            acc = sum((h-ytest)==0)/length(ytest);
            y_hat = 1./ (1+ exp(-(Xte*w_train+c_train) ) );
            y_true = ytest;
            
            
            total_acc(i,j,k) = mean(acc);                       % accuracy 
            Y_hat{i,j,k} = y_hat;
            Y_true{i,j,k} = y_true;
            
            [~,~,~,A1(i,j,k)] = perfcurve(ytest,Xte,1);         % AUC based on features and labels
            [~,~,~,A2(i,j,k)] = perfcurve(ytest,y_hat,1);       % AUC based on LOSO classifier and labels
            
        end
    end
end

clear X
clear Xtrain
clear Xi
clear Xtr
clear Xte
clear ica_a



condname = [];
for c = 1:numConds
    condname = [condname,conds{c}];
end




% save the results
resultpath = '/mnt/bucket/labs/norman/boyuw/results/Results2017_07_31/LR/partial_ICAL1_Average/';
mkdir(resultpath);
save([resultpath,'/LR_',filename,'_',condname,'_',num2str(job_id),'_',num2str(so)]);

