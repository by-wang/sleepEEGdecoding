function replayL1classify_withSub(job_id)

% Replay decoding using l1-regularized logistic regression for each
% subject, with leave-one-trial-out cross-validation. 
% We use the logistic regression implementation from SLEP toolbox:
% http://www.yelab.net/software/SLEP/



%--------

% add auxiliary functions for EEG processing and analysis
addpath(genpath('/mnt/bucket/people/boyuw/code/utilities'));



params.trial = 'original';                                                 % we use the original voltage signals
params.idx = [];

% the replay performances we try to classify
params.cond1 = '11';                                                       
params.cond2 = '10';


% No pre-processing step applied. 
% Other options: 
% rescale - rescale the  signals such that the range is [-1, 1]; 
% normal - normalize the the signals such that it has 0 mean and 1 std.
prepro = 'none';                                                           



% parameters for l1-regularized logistic regression. For more details, see
% the SLEP manual 

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




% read the task ID for parallel computing, so is the subject ID
SGE_TASK_ID = str2double(getenv('SLURM_ARRAY_TASK_ID'));%SLURM
so = SGE_TASK_ID;   


datapath = '/mnt/bucket/labs/norman/jantony/motorExp/2_23_2017';

list = dir(datapath);
if strcmp(list(3).name,'.DS_Store')
    list(1:3) = [];
else
    list(1:2) = [];
end




startpoint = -2;                        % start point of EEG signals                            
width = 7;                              % length of the signals
rate = 1;                               % undersampling rate. 1 = no undersampling


    
   
    
% Initialize the parameters for EEG signals extraction            
opts_eeg = optsInit_eeg(rate,startpoint,width);
opts_eeg.cond = params.cond1;           % cond1 = '11' or '00';
opts_eeg.trial = params.trial;          % voltage/rms/rmst
opts_eeg.bslc = 0;                      % baseline correction or not
opts_eeg.idx = params.idx;              % if extract the signals from specific channels. if empty, then use all channels
opts_eeg.Num = [];



% Extract EEG signals
dict = [datapath '/' list(so).name '/sleep/ERP_4ms.mat'];
filename = 'unwrap';        
[X1,~] = eegPrepare(dict,filename,opts_eeg);
N1 = size(X1,1);
y1 = -ones(N1,1);

opts_eeg.cond = params.cond2;           % cond2 = '10' or '01';
[X2,~] = eegPrepare(dict,filename,opts_eeg);
N2 = size(X2,1);
y2 = ones(N2,1);


Xtmp = [X1;X2];

% Pre-processing
if strcmp(prepro,'rescale') 

    Nt = size(Xtmp,1);                  % number of trials 
    Nc = size(Xtmp,3);                  % number of channels

    for t = 1:Nt
        xtmp = squeeze(Xtmp(t,:,:));
        for c = 1:Nc
            xxtmp = xtmp(:,c);
            xmax = max(xxtmp);
            xmin = min(xxtmp);
            xtmp(:,c) = (2*xxtmp-xmax-xmin)/(xmax-xmin);
        end
        Xtmp(t,:,:) = xtmp;
    end
    
elseif strcmp(prepro,'normal') 

    Nt = size(Xtmp,1);
    Nc = size(Xtmp,3);

    for t = 1:Nt
        xtmp = squeeze(Xtmp(t,:,:));
        for c = 1:Nc
        xxtmp = xtmp(:,c);
            xmean = mean(xxtmp);
            xstd = std(xxtmp);
            xtmp(:,c) = (xxtmp-xmean)/xstd;
        end
        Xtmp(t,:,:) = xtmp;
    end
end
        
X = Xtmp;
Y = [y1;y2];

    
   

clear X1 X2;
clear y1 y2;
clear Xtmp

    
    
% regularization parameters
rho = [0.01,0.001,0.0005,0.00001];
Nr = length(rho);


            
       
[~, numT, ~] = size(X);                 % number of time points

width = [1,25,50,125,250];              
numW = length(width);                   % number of window lengths


total_acc = zeros(numT,numW,Nr);
Y_hat = cell(numT,numW,Nr);
Y_true = cell(numT,numW,Nr);

       
numTrial = size(X,1);



% Leave-one-trial-out cross-validation at different time points, with
% different window lengths and regularization parameters   
for i = 1:numW
    wid = width(i);
    for j = 1:numT


        endIdx = min(j+wid-1,numT);
        
        
        Xtmp = X(:,j:endIdx,:);
        Xtmp = squeeze(mean(Xtmp,2));                       % Average over time points


            for r = 1:Nr
                acc = nan(numTrial,1);
                y_hat = nan(numTrial,1);
                y_true = nan(numTrial,1);
                
                for tt = 1:numTrial                         % each time take one trial out for testing                      
    
                    Xte = Xtmp(tt,:);
                    ytest = Y(tt);

                    Xtr = Xtmp;
                    ytrain = Y;

                    Xtr(tt,:) = [];
                    ytrain(tt) = [];
                
                
                
                    % train a logistic regression classifier 
                    [w_train, c_train, ~, ~]= LogisticR(Xtr, ytrain, rho(r), opts);
                    
                    % make prediction
                    h = sign(Xte*w_train+c_train);
                    acc(tt) = sum((h-ytest)==0)/length(ytest);
                    y_hat(tt) = 1./ (1+ exp(-(Xte*w_train+c_train) ) );
                    y_true(tt) = ytest;
                end


                total_acc(j,i,r) = mean(acc);
                Y_hat{j,i,r} = y_hat;
                Y_true{j,i,r} = y_true;
            end
            clear Xtmp
   end
end



           
            
clear X
clear Y
clear Xtrain
clear Xtest
clear ytrain
clear ytest
clear Ytmp
clear Xtr
clear Xte
            

if opts_eeg.bslc == 1
    resultpath = '/mnt/bucket/labs/norman/boyuw/results/Results2017_05_15/Motor/L1/withinSub/wbslc';
else
    resultpath = '/mnt/bucket/labs/norman/boyuw/results/Results2017_05_15/Motor/L1/withinSub/wobslc';
end

% save the results
mkdir(resultpath);
save([resultpath,'/',num2str(job_id),'_',params.cond1,'vs',params.cond2,'_',prepro,'_',params.trial,'_',list(so).name]);

    
            



