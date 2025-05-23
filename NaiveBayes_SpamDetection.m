%% Naive Bayes Spam Detection Project - Full Real World Version
% Dataset: Spambase | Enhancements: Feature pruning, correlation removal, model comparison, class imbalance, bootstrap CI, ROC, PR curve, deployment

%% 0. Initialization
clc; clear; close all;
rng(1); % For reproducibility

%% 1. Load Dataset
disp('Loading Spambase dataset...');
data = readtable('spambase.csv', 'VariableNamingRule', 'preserve');

X = data{:,1:end-1};
y = data{:,end};

fprintf('Original features: %d\n', size(X,2));

%% 2. Remove Low-Variance and Highly Correlated Features
varThreshold = 0.01;
lowVarIdx = var(X) < varThreshold;
X(:, lowVarIdx) = [];
fprintf('Removed %d low-variance features.\n', sum(lowVarIdx));

% Remove highly correlated features
R = corrcoef(X);
upperTri = triu(R,1);
[rows, cols] = find(abs(upperTri) > 0.95);
colsToRemove = unique(cols);
X(:, colsToRemove) = [];
fprintf('Removed %d highly correlated features.\n', numel(colsToRemove));

%% 3. Visualize Class Distribution
figure;
histogram(y, 'BinMethod', 'integers');
xticks([0 1]); xticklabels({'Non-Spam', 'Spam'});
xlabel('Email Type'); ylabel('Number of Emails');
title('Class Distribution in Spambase Dataset');
grid on;

%% 4. Train-Test Split (Stratified)
disp('Splitting data into training and testing sets...');
cvp = cvpartition(y, 'Holdout', 0.3);
Xtrain = X(training(cvp), :);
ytrain = y(training(cvp), :);
Xtest = X(test(cvp), :);
ytest = y(test(cvp), :);
disp('Data split complete.');

%% 5. Train Naive Bayes with Hyperparameter Tuning
disp('Training Naive Bayes model with hyperparameter optimization...');
nbModel = fitcnb(Xtrain, ytrain, ...
    'DistributionNames', 'kernel', ...
    'OptimizeHyperparameters', {'Kernel','Width'}, ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 30));
disp('Model training and optimization complete.');

%% 6. Cross-Validation Results from Optimization
cvLoss = nbModel.HyperparameterOptimizationResults.MinObjective;
cvAcc = (1 - cvLoss) * 100;
fprintf('Cross-Validated Accuracy from Tuning: %.4f%%\n', cvAcc);

%% 7. Retrain Final Model with Best Hyperparameters
bestParams = nbModel.HyperparameterOptimizationResults.XAtMinObjective;
retrainArgs = {'DistributionNames', 'kernel'};
if ismember('Kernel', bestParams.Properties.VariableNames)
    retrainArgs = [retrainArgs, {'Kernel', char(bestParams.Kernel)}];
end
if ismember('Width', bestParams.Properties.VariableNames)
    retrainArgs = [retrainArgs, {'Width', bestParams.Width}];
end
finalModel = fitcnb(Xtrain, ytrain, retrainArgs{:});
disp('Final model retrained.');

%% 8. 5-Fold Cross-Validation on Final Model
cvFinal = crossval(finalModel, 'KFold', 5);
cvLossFinal = kfoldLoss(cvFinal);
cvAccFinal = (1 - cvLossFinal) * 100;
fprintf('Final 5-Fold CV Accuracy: %.4f%%\n', cvAccFinal);

%% 9. Predict on Test Set
[ypred, scores] = predict(finalModel, Xtest);
testAccuracy = mean(ypred == ytest) * 100;
fprintf('Test Set Accuracy: %.4f%%\n', testAccuracy);

%% 10. Evaluate Test Performance
C = confusionmat(ytest, ypred);
TP = C(2,2); TN = C(1,1); FP = C(1,2); FN = C(2,1);
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1 = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.5f\n', precision);
fprintf('Recall: %.5f\n', recall);
fprintf('F1-Score: %.5f\n', f1);

%% 11. Bootstrap Confidence Interval for Accuracy
bootAcc = bootstrp(1000, @(idx) mean(ypred(idx) == ytest(idx)), 1:length(ytest));
ci = prctile(bootAcc, [2.5 97.5]);
fprintf('Bootstrap 95%% CI for Test Accuracy: [%.4f, %.4f]\n', ci(1)*100, ci(2)*100);

%% 12. Confusion Matrix Plot
figure;
confusionchart(ytest, ypred, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Confusion Matrix: Naive Bayes with Smoothing');

%% 13. ROC Curve and AUC
[Xroc, Yroc, ~, AUC] = perfcurve(ytest, scores(:,2), 1);
figure;
plot(Xroc, Yroc, 'LineWidth', 2); hold on;
plot([0 1], [0 1], 'k--'); hold off;
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve for Naive Bayes (AUC = %.5f)', AUC));
grid on;

%% 14. Precision-Recall Curve
[prec, rec, ~, ~] = perfcurve(ytest, scores(:,2), 1, 'xCrit', 'reca', 'yCrit', 'prec');
figure;
plot(rec, prec, 'b-', 'LineWidth', 2);
xlabel('Recall'); ylabel('Precision');
title('Precision-Recall Curve');
grid on;

%% 15. Compare with Logistic Regression
logModel = fitclinear(Xtrain, ytrain, 'Learner', 'logistic');
ypredLog = predict(logModel, Xtest);
accLog = mean(ypredLog == ytest) * 100;
fprintf('Logistic Regression Test Accuracy: %.4f%%\n', accLog);

%% 16. Save Final Model for Deployment
saveLearnerForCoder(finalModel, 'SpamClassifierModel');
disp('Final model saved to SpamClassifierModel.mat');

%% 17. Final Summary
disp('--- Final Summary ---');
disp(['Cross-Validated Accuracy from Tuning: ', num2str(cvAcc), '%']);
disp(['Final 5-Fold CV Accuracy: ', num2str(cvAccFinal), '%']);
disp(['Test Set Accuracy: ', num2str(testAccuracy), '%']);
disp(['Precision: ', num2str(precision)]);
disp(['Recall: ', num2str(recall)]);
disp(['F1-Score: ', num2str(f1)]);
disp(['AUC: ', num2str(AUC)]);
disp(['95% CI for Test Accuracy: [', num2str(ci(1)*100), ', ', num2str(ci(2)*100), ']']);
disp(['Logistic Regression Test Accuracy: ', num2str(accLog), '%']);
