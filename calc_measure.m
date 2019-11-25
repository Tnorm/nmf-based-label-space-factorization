function [all_measures] = calc_measure(Y, Y_hat)
% Y : N * label_num,    ground-truth labelling matrix
% Y_hat : N * label_num,     predicted labelling matrix
% performance metrics: label-based macro_F1, label-based micro_F1 & example-based accuracy

% in case the format is not correct
Y(Y<=0) = 0;
Y(Y>0) = 1;
Y_hat(Y_hat<=0) = 0;
Y_hat(Y_hat>0) = 1;

% label-based measures:
% calculate macro_averaged meaures
[~, label_num] = size(Y);
accus = zeros(1, label_num);
precs = zeros(1, label_num);
recs = zeros(1, label_num);
F1s = zeros(1, label_num);
TPs = sum(Y  &  Y_hat);
FPs = sum(~Y  &  Y_hat);
TNs = sum(~Y  &  ~Y_hat);
FNs = sum(Y  &  ~Y_hat);

for j = 1 : label_num,
    if(TPs(j) + FPs(j) + TNs(j) + FNs(j) > 0)
        accus(1, j) = (TPs(j) + TNs(j)) / (TPs(j) + FPs(j) + TNs(j) + FNs(j));
    else
        accus(1, j) = 0;
    end
    
    if (TPs(j) + FPs(j) > 0)
        precs(1, j) = TPs(j) / (TPs(j) + FPs(j));
    else
        precs(1, j) = 0;
    end
    
    if (TPs(j) + FNs(j) > 0)
        recs(1, j) = TPs(j) / (TPs(j) + FNs(j));
    else
        recs(1, j) = 0.5;
    end
    
    if precs(1, j) + recs(1, j) == 0
        F1s(1, j) = 0;
    else
        F1s(1, j) = 2 * precs(1, j) * recs(1, j) / (precs(1, j) + recs(1, j));
    end
end
macro_accuracy = mean(accus);
macro_recall = mean(recs);
macro_precision = mean(precs);
macro_F1 = mean(F1s);

% calculate micro-averaged measures
TP = sum(TPs);
FP = sum(FPs);
TN = sum(TNs);
FN = sum(FNs);
if(TP + FP + TN + FN > 0)
    micro_accuracy = (TP + TN) / (TP + FP + TN + FN);
else
    micro_accuracy = 0;
end

if (TP + FP> 0)
    micro_precision = TP / (TP + FP);
else
    micro_precision = 0;
end

if (TP + FN > 0)
    micro_recall = TP / (TP + FN);
else
    micro_recall = 0.5;
end

if (micro_precision + micro_recall == 0)
    micro_F1 = 0;
else
    micro_F1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall);
end


% example-based measures:
% calculate accuracy
accuracy = sum(Y & Y_hat, 2) ./ sum(Y | Y_hat, 2);
accuracy(isinf(accuracy)) = 0.5;
accuracy(isnan(accuracy)) = 0.5;
accuracy_exam = mean(accuracy);

% calculate F1-exam
precs_exam = sum(Y & Y_hat, 2) ./ sum(Y_hat, 2);
precs_exam(isinf(precs_exam)) = 0;
precs_exam(isnan(precs_exam)) = 0;
recs_exam = sum(Y & Y_hat, 2) ./ sum(Y, 2);
recs_exam(isinf(recs_exam)) = 0.5;
recs_exam(isnan(recs_exam)) = 0.5;
F1s_exam = 2 * (precs_exam .* recs_exam) ./ ( precs_exam + recs_exam);
F1s_exam(isinf(F1s_exam)) = 0;
F1s_exam(isnan(F1s_exam)) = 0;
F1_exam = mean(F1s_exam);
recall_exam = mean(recs_exam);
precision_exam = mean(precs_exam);

% ready all_measures
all_measures = zeros(12, 1);
all_measures(1) = macro_accuracy;
all_measures(2) = micro_accuracy;
all_measures(3) = macro_recall;
all_measures(4) = micro_recall;
all_measures(5) = macro_precision;
all_measures(6) = micro_precision;
all_measures(7) = macro_F1;
all_measures(8) = micro_F1;
all_measures(9) = accuracy_exam;
all_measures(10) = recall_exam;
all_measures(11) = precision_exam;
all_measures(12) = F1_exam;

end


% function [macro_accuracy, micro_accuracy, macro_recall, micro_recall, macro_precision, micro_precision, macro_F1, micro_F1,      accuracy_exam, recall_exam, precision_exam, F1_exam] = calc_measure_good(Y, Y_hat)
