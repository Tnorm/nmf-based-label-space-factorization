%% Initialize Data

dbSelect = 3;
kernel = 1;


if dbSelect == 1
    dName = 'cal500';
    A = load(['Datasets/' dName '.mat']);
    numberOfTrains = 502;
%     betaBest = 0.7;
%     latentSizeBest = 14;
%     gamaBest = 0.001;
    betaBest = 0.7;
    latentSizeBest = 24;
    gamaBest = 0.001;
end

if dbSelect == 2
    dName = 'enron';
    A = load(['Datasets/' dName '.mat']);
    numberOfTrains = 1702;
%     betaBest = 0.9;
%     latentSizeBest = 12;
%     gamaBest = 0.001;
    betaBest = 0.9;
    latentSizeBest = 12;
    gamaBest = 0.001;
end

if dbSelect == 3
    dName = 'corel5k';
    A = load(['Datasets/lear_DenseHue__' dName '.mat']);
    numberOfTrains = 4999;
    betaBest = 0.7;
    latentSizeBest = 14;
    gamaBest = 0.0005;
%     betaBest = 0.7;
%     latentSizeBest = 14;
%     gamaBest = 0.0005;
end

if dbSelect == 4
    dName = 'iaprtc12';
    A = load(['Datasets/lear_DenseHue__' dName '.mat']);
    numberOfTrains = 8000;
%     betaBest = 0.3;
%     latentSizeBest = 27;
%     gamaBest = 0.001;
    betaBest = 0.3;
    latentSizeBest = 12;
    gamaBest = 0.001;
end

if dbSelect == 5
    dName = 'espgame';
    A = load(['Datasets/lear_DenseHue__' dName '.mat']);
    numberOfTrains = 11000;
    betaBest = 0.3;
    latentSizeBest = 14;
    gamaBest = 0.0005;
end

if dbSelect == 6
    dName = 'mediamill';
    A = load(['Datasets/' dName '.mat']);
    numberOfTrains = 12000;
    betaBest = 0.9;
    latentSizeBest = 14;
    gamaBest = 0.001;
end



% A = load('Datasets/delicious.mat');




size(A.Y_all)
randrows = randperm(size(A.Y_all(1:numberOfTrains,:),1));

X = A.X_all(randrows,:);
Y = A.Y_all(randrows,:);

BigMat = [];

%% Define Parameters ------------------------------
%--------------------------------------------------
%--------------------------------------------------
%--------------------------------------------------
%--------------------------------------------------

% sigma = 1000;


N = size(X,1); %number of samples
d = size(X,2); %number of features

gama =0.3;
alpha = 1;

% numOfLabels = size(Y,2);
% latent_size = floor(numOfLabels/2);
sss = 0;

% latents = [0.01 0.1 0.2 0.3 0.4 0.5 0.6];
% for latent_size = ceil(latents*d)
% for eng = 0.0005
% for beta = 0.3

for latent_size = latentSizeBest;
for eng = gamaBest;
for beta = 2;

% for beta = 2;

% for latent_size = 8:2:16
% for eng = [0.0005 0.001]
% for beta = 0.3:0.1:0.9

sss = sss + 1;
sss
numOfFolds = 10;
sizeOfFolds = floor(N/numOfFolds);
all_indices = 1:N;

numberOfIters = 500;

threshold = .5;

% Initial values for Evaluation Measures in K-fold-Cross-Validation
mf1 = 0;
ea = 0;
ef1 = 0;
%% Training

% %% Create dataset chunks
% for k = 1:numOfFolds
%     % K Fold Cross Validation
%     [X_train, X_test, Y_train, Y_test] = cross_val(X,Y,k,all_indices,sizeOfFolds);
%     if ~exist(['datasets_chunks/' dName], 'dir')
%         mkdir(['datasets_chunks/' dName]);
%     end
%     save(['datasets_chunks/' dName '/' dName '_' num2str(k)], 'X_train', 'X_test', 'Y_train', 'Y_test');
%     BB = load(['datasets_chunks/' dName '/' dName '_' num2str(k)]);
% end



for k = 1:numOfFolds
    % K Fold Cross Validation
    [X_Train, X_Test, Y_Train, Y_Test] = cross_val(X,Y,k,all_indices,sizeOfFolds);
    
%     [X_Train, X_Test] = ak_rpSVD(X_Train,X_Test,1000);

    k
    C_old = rand(size(X_Train,1),latent_size);
    D_old = rand(latent_size, size(Y_Train,2));

    if kernel
        sigma = ak_large_datasets_get_proper_rbf_sigma(X_Train, size(X_Train,1));
%         sigma = sigma /2;
        K = ak_fast_cross_rbf_kernel(X_Train,X_Train,sigma);
%           K = X_Train * X_Train';
%         [K sigma] = calc_kernel(X_Train);
%         K = rbf(X_Train, sigma);
%         [K sigma] = calc_kernel(X_Train);
%         K = K + eye(size(K,1))*0.0001;
    end
    
    if kernel
%         PX = K*((K'*K + eye(size(K,1))*0.0001)^-1)*K';
        PX = eye(size(K,1));
    else
        PX = X_Train*((X_Train'*X_Train)^-1)*X_Train';
    end

%     PXX = PX' * PX;
    
    for iter=1:numberOfIters
        
        C_new = C_old + eng*(2*alpha*PX*C_old - 2*beta*C_old*(D_old*D_old')+ 2*beta*Y_Train*D_old' - 2*alpha*C_old); 
        C_new(find(C_new<0))=0;
        D_new = D_old + eng*(-2*beta*(C_old'*C_old)*D_old + 2*beta*C_old'*Y_Train+ 2*gama*D_old);
        D_new(find(D_new<0))=0;
        
        D_old = D_new;
        C_old = C_new;
        %iter
    end
    
    D = D_new;
    C = C_new;
    
%     [C D] = nnmf(Y_Train, latent_size,'algorithm','als');

    if kernel
        lala = 0.0001;
        ALPH = (K + lala*eye(size(K,1)))^-1*C;
    else
        Wreg = (X_Train'*X_Train)^-1*X_Train'*C;
    end
    
%% Testing
    if kernel
        TDis = ak_fast_cross_rbf_kernel(X_Test,X_Train,sigma);
%         TDis = X_Test * X_Train';
%         Y_hat = (TDis*((K)^-1)'*C)*D;
        Y_hat = (TDis*ALPH)*D;
    else
        newC = X_Test*Wreg;
%         newC(find(newC<0))=0;
        Y_hat = newC*D;
    end
    
    Y_hat(find(Y_hat> threshold))=1;
    Y_hat(find(Y_hat<= threshold))=0;
    
    results = calc_measure(Y_Test, Y_hat);
    %auc = ak_auc_tp_fp_diffrent_ks(Y_hat, Y_Test);
    
    
%% Print Results
    disp('micro_f1')
    results(8)
    disp('example_accuracy')
	results(9)
    disp('example_F1')
    results(12)
      
    mf(k) = results(8);
    ea(k) = results(9);
    ef(k) = results(12);
end


%% Print Overall Results
%disp('mean_micro_f1')
%mf1 / numOfFolds
%disp('mean_example_accuracy')
%ea / numOfFolds
%disp('mean_example_F1')
%ef1 / numOfFolds
%S 
BigMat = [BigMat; alpha beta gama eng latent_size sum(mf)/numOfFolds sqrt(var(mf)) sum(ea)/numOfFolds sqrt(var(ea)) sum(ef)/numOfFolds sqrt(var(ef))]; %auc];

end
end
end


means = (BigMat(:,6) + BigMat(:,8) + BigMat(:,6))/3;
maxMeans = find(means == max(means));
bestRow = BigMat(maxMeans,:); 

BigMat = [BigMat; zeros(size(bestRow)) ; bestRow];

csvwrite('Results.csv', BigMat);


%xlswrite('Results.xlsx', BigMat);
