function [Y_hat,y_true] = loadReplay(replaypath,list,SubIdx)

% This function returns the output of replay classifier given the path of
% the results and the subject ID.

numSub = length(list);

Y_hat = [];
y_true = [];

for i = 1:numSub
    SubName = list(i).name(end-6:end-4);
    idx = str2num(SubName);
    if idx == SubIdx
        dict = [replaypath '/' list(i).name];
        load(dict,'Y_hat','y_true');
        break;    
    end
end
