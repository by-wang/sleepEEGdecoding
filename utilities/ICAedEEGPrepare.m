function [X, y] = ICAedEEGPrepare(X,y,ref,conds)

% This functiion returns the ICAed EEG signals with different conditions
% (i.e., pre/post-nap performances)



% the left/right reference fields, where the 4-th column is the indicator 
% pre-nap performance, the 5-th column is the indicator of post-nap 
% performance
ref1 = ref{1}.ref;                  
ref2 = ref{2}.ref;


num1 = size(ref1,1);
num2 = size(ref2,1);

X1 = X(1:num1,:);
X2 = X(num1+1:num1+num2,:);

y1 = y(1:num1);
y2 = y(num1+1:num1+num2);



if strcmp(conds,'11') 
    idx1 = find(ref1(:,4) == 1 & ref1(:,5) ==1);
    idx2 = find(ref2(:,4) == 1 & ref2(:,5) ==1);
elseif strcmp(conds,'01') 
    idx1 = find(ref1(:,4) == 0 & ref1(:,5) ==1);
    idx2 = find(ref2(:,4) == 0 & ref2(:,5) ==1);
elseif strcmp(conds,'10') 
    idx1 = find(ref1(:,4) == 1 & ref1(:,5) == 0);
    idx2 = find(ref2(:,4) == 1 & ref2(:,5) == 0);
elseif strcmp(conds,'00') 
    idx1 = find(ref1(:,4) == 0 & ref1(:,5) == 0);
    idx2 = find(ref2(:,4) == 0 & ref2(:,5) == 0);
end


X1 = X1(idx1,:);
X2 = X2(idx2,:);

y1 = y1(idx1);
y2 = y2(idx2);


X = [X1;X2];
y = [y1;y2];