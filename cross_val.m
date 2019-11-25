function [X_Train, X_Test, Y_Train, Y_Test] =  cross_val(X,Y, k,all_indices,sizeOfFolds)

    test_indexes = (k-1)*sizeOfFolds+1:k*sizeOfFolds;
    train_indexes = all_indices;
    train_indexes(test_indexes) = [];
    
    X_Test = X(test_indexes,:);
    X_Train = X(train_indexes,:);

    Y_Test = Y(test_indexes,:);
    Y_Train = Y(train_indexes,:);