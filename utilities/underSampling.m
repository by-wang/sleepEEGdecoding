function Z = underSampling(X,rate,method)    

% This function returns the undersampled EEG signals given the
% undersampling rate

if rate == 1
    Z = X;
    
elseif rate == 0
    Z = squeeze(mean(X,2));
    if size(X,1) == 1
        Z = Z';
    end
else
    
    [numT, numS, numC] = size(X);                                          % number of trials/time points/channels
    nS = floor(numS/rate);                                                 % for simplicity, if mod(numS,rate)~=0, we discard the last points
    Z = zeros(numT,nS,numC);
    

    for k = 1:nS            
        if strcmp(method,'mean')
            Z(:,k,:) = mean(X(:,rate*(k-1)+1:k*rate,:),2);
        elseif strcmp(method,'max')
            Z(:,k,:) = max(X(:,rate*(k-1)+1:k*rate,:),[],2);
        end
    end
end
            
        
