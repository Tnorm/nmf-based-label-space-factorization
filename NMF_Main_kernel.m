%% Initialize Data

%A = load('Datasets/lear_DenseHue__iaprtc12.mat');
%  A = load('Datasets/lear_DenseHue__corel5k.mat');
% A = load('Datasets/lear_DenseHue__espgame.mat');



numberOfTrains = 4999;

randrows = randperm(size(A.Y_all(1:numberOfTrains,:),1));

X = A.X_all(randrows,:);
Y = A.Y_all(randrows,:);

BigMat = [];

%% Define Parameters
kernel = 1;

N = size(X,1);

% numOfLabels = size(Y,2);
% latent_size = floor(numOfLabels/2);

for latent_size = 18
for eng = 0.0005:0.0005
for alpha = 1:1
for beta = 0.3:0.3
for gama = 0.1:0.1
%latent_size = 20;
%eng = 0.0005;
%alpha = 1;
%beta = 0.4;
%gama = 0.5;

numOfFolds = 10;
sizeOfFolds = floor(N/numOfFolds);
all_indices = 1:N;

numberOfIters = 200;

threshold = .5;

% Initial values for Evaluation Measures in K-fold-Cross-Validation
mf1 = 0;
ea = 0;
ef1 = 0;
%% Training

for k = 1:numOfFolds
    
    % K Fold Cross Validation
    [X_Train, X_Test, Y_Train, Y_Test] = cross_val(X,Y,k,all_indices,sizeOfFolds);
    
%     [X_Train, X_Test] = ak_rpSVD(X_Train,X_Test,1000);

    k
    C_old = rand(size(X_Train,1),latent_size);
    D_old = rand(latent_size, size(Y_Train,2));
    if kernel
        sigma = 150;
        K = rbf(X_Train, sigma);
        %[K sigma] = calc_kernel(X_Train);
        K = K + eye(size(K,1))*0.0001;
    end
    
    if kernel
        PX = K*((K'*K)^-1)*K';
    else
        PX = X_Train*((X_Train'*X_Train)^-1)*X_Train';
    end
    %PXX = PX' * PX;
    
    for iter=1:numberOfIters
        
        C_new = C_old + eng*(2*alpha*PX*C_old - 2*beta*C_old*(D_old*D_old')+ 2*beta*Y_Train*D_old' -2*C_old); 
        C_new(find(C_new<0))=0;
        
        D_new = D_old + eng*(-2*beta*(C_old'*C_old)*D_old + 2*beta*C_old'*Y_Train+ 2*gama*D_old);
        D_new(find(D_new<0))=0;
        
        D_old = D_new;
        C_old = C_new;
        iter
    end
    
    D = D_new;
    C = C_new;
    if kernel
        lala = 0;
        ALPH = (K + lala*eye(size(K,1)))^-1*C;
    else
        Wreg = (X_Train'*X_Train)^-1*X_Train'*C;
    end

%% Testing
    
    if kernel
        TDis = ak_fast_cross_rbf_kernel(X_Test,X_Train,sigma);
        Y_hat = (C'*((K)^-1)*TDis')'*D;
        %Y_hat = (TDis*ALPH)*D;
    else
        Y_hat = (Wreg'*X_Test')'*D;
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
    
   
    mf1 = mf1 +  results(8);
    ea = ea + results(9);
    ef1 = ef1 + results(12);
end

%% Print Overall Results
%disp('mean_micro_f1')
%mf1 / numOfFolds
%disp('mean_example_accuracy')
%ea / numOfFolds
%disp('mean_example_F1')
%ef1 / numOfFolds
%S 
BigMat = [BigMat; alpha beta gama eng latent_size mf1/numOfFolds ea/numOfFolds ef1/numOfFolds]; %auc];

end
end
end
end
end

csvwrite('Results.csv', BigMat);
%xlswrite('Results.xlsx', BigMat);