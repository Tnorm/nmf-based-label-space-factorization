%% Initialize Data

% A = load('Datasets/lear_DenseHue__iaprtc12.mat');
 %A = load('Datasets/lear_DenseHue__corel5k.mat');
% A = load('Datasets/lear_DenseHue__espgame.mat');
A = load('Datasets/cal500');



numberOfTrains = 502;

randrows = randperm(size(A.Y_all(1:numberOfTrains,:),1));

X = A.X_all(randrows,:);
Y = A.Y_all(randrows,:);

%% Define Parameters
kernel = 1;
N = size(X,1);

% numOfLabels = size(Y,2);
% latent_size = floor(numOfLabels/2);

latent_size = 10;

eng = 0.0005;
alpha = 1;
beta = 0.9;
gama = 0.01;


numOfFolds = 10;
sizeOfFolds = floor(N/numOfFolds);
all_indices = 1:N;

numberOfIters = 30;

threshold = 0.6;

% Initial values for Evaluation Measures in K-fold-Cross-Validation
mf1 = 0;
ea = 0;
ef1 = 0;
%% Training

for k = 1:numOfFolds
    
    % K Fold Cross Validation
    [X_Train, X_Test, Y_Train, Y_Test] = cross_val(X,Y,k,all_indices,sizeOfFolds);
    
    one_Y = mean(sum(Y_Train, 2));
    zero_Y = size(Y_Train,2) - one_Y;
    mu = zero_Y / one_Y;
    
%     [X_Train, X_Test] = ak_rpSVD(X_Train,X_Test,1000);

    k
    C_old = rand(size(X_Train,1),latent_size);
    D_old = rand(latent_size, size(Y_Train,2));


    if kernel
        sigma = ak_large_datasets_get_proper_rbf_sigma(X_Train, size(X_Train,1));
        sigma = 1000;
        K = ak_fast_cross_rbf_kernel(X_Train,X_Train,sigma);
    end
    
    if kernel
        PX = eye(size(K,1));
    else
        PX = X_Train*((X_Train'*X_Train)^-1)*X_Train';
    end
    
    for iter=1:numberOfIters
        
    Tmp = sigmoid(C_old*D_old, beta);
    CT = mu*Y_Train.*(1-Tmp) - (1-Y_Train).*Tmp;
    DrC = zeros(size(C_old,1), size(C_old,2));
        for i=1:size(C_old,1)
            for k=1:size(C_old,2)
                %for j =1:size(D_old,2)
                %    DrC(i,k) = DrC(i,k) - D_old(k,j)*CT(i,j);
                %end
                DrC(i,k) = sum(D_old(k,:).*CT(i,:))*-1;
            end
        end
        
    DrD = zeros(size(D_old,1), size(D_old,2));
        for k=1:size(D_old,1)
            for j=1:size(D_old,2)
                %for i =1:size(C_old,1)
                %    DrD(k,j) = DrD(k,j) - C_old(i,k)*CT(i,j);
                %end
                DrD(k,j) = sum(C_old(:,k).*CT(:,j))*-1;
            end
        end
    C_new = C_old - eng*(2*alpha*PX*C_old + beta*DrC);
    C_new(find(C_new<0))=0;
    D_new = D_old - eng*(beta*DrD + 2*gama*D_old);
    D_new(find(D_new<0))=0;
        D_old = D_new;
        C_old = C_new;
        iter
    end
    
    D = D_new;
    C = C_new;
    
    if kernel
        lala = 0.0001;
        ALPH = (K + lala*eye(size(K,1)))^-1*C;
    else
        Wreg = (X_Train'*X_Train)^-1*X_Train'*C;
    end

%% Testing
    
    if kernel
        TDis = ak_fast_cross_rbf_kernel(X_Test,X_Train,sigma);
        Y_hat = (TDis*ALPH)*D;
    else
        newC = X_Test*Wreg;
        Y_hat = newC*D;
    end

    
    Y_hat(find(Y_hat> threshold))=1;
    Y_hat(find(Y_hat<= threshold))=0;
    
    results = calc_measure(Y_Test, Y_hat);
    
    
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
disp('mean_micro_f1')       
mf1 / numOfFolds
disp('mean_example_accuracy')
ea / numOfFolds
disp('mean_example_F1')
ef1 / numOfFolds


