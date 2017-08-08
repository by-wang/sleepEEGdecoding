function opts = optsInit_eeg(rate,startpoint,width)
% This function initialize the the parameters for EEG signals extraction

opts.idx = [];
opts.s = 250;                                       % sampling rate
opts.unit = 1000;                                   % unit (millisecond) used for the recordings  
opts.offset = startpoint;                           % start point
opts.width = width;                                 % signal length
opts.rate = rate;                                   % undersampling rate
opts.method = 'mean';