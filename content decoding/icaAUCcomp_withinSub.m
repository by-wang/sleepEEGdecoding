function icaAUCcomp_withinSub(job_id)

% This function computes the AUC values of each ICA component for content 
% decoding (left/right) using within subject (withinSub) classifier. Each
% time it takes one subject, leaves one trial out as the testing sample,
% using the rest trials as the training samples to train a logistic
% regression classifier



%--------

% add auxiliary functions for EEG processing and analysis
addpath(genpath('/mnt/bucket/people/boyuw/code'));


% pre/post-nap performances. the first digit encodes the pre-nap 
% performance, the second digit encodes the post-nap performance. 
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



% Xi = cell(1,numSub);
% Yi = cell(1,numSub);


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
    
    N = length(yica);               % number of trials
    
    
    for j = 1:numW
        wid = width(j);
        for k = 1:numT
            endIdx = min(k+wid-1,numT);
            
            Xtemp = Xica(:,k:endIdx);
            Xtemp = mean(Xtemp,2);
            
            y_hat = nan(N,1);
            h = nan(N,1);
            y_true = nan(N,1);
            acc = nan(N,1);
            for n = 1:N
                Xtrain = Xtemp;
                Xtrain(n) = [];
                Xtest = Xtemp(n);
                
                ytrain = yica;
                ytrain(n) = [];
                ytest = yica(n);
            
%             Xte = Xtest(:,k:endIdx);
%             Xte = mean(Xte,2);
                numPos = length(find(ytrain == 1));
                numNeg = length(find(ytrain == -1));
                opts.sWeight(1) = numNeg/(numPos+numNeg);
                opts.sWeight(2) = numPos/(numPos+numNeg);
                [w_train, c_train, ~, ~]= LogisticR(Xtrain, ytrain, 0.000000001, opts);    % training

                h(n) = sign(Xtest*w_train+c_train);                                          % testing
                acc(n) = sum((h-ytest)==0)/length(ytest);
                y_hat(n) = 1./ (1+ exp(-(Xtest*w_train+c_train) ) );
                y_true(n) = ytest;
            end
            
            
            total_acc(i,j,k) = mean(acc);                       % accuracy 
            Y_hat{i,j,k} = y_hat;
            Y_true{i,j,k} = y_true;
            
            [~,~,~,A1(i,j,k)] = perfcurve(y_true,Xtemp,1);         % AUC based on features and labels
            [~,~,~,A2(i,j,k)] = perfcurve(y_true,y_hat,1);       % AUC based on LOSO classifier and labels
            
        end
    end
    
    

end

clear Xica
clear yica




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

