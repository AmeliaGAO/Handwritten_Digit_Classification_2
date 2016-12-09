%% Clear workspace.
clear all;
close all;

%% Load data.
load('digits.mat');
train_inputs  = [train2.';train3.'];
valid_inputs  = [valid2.';valid3.'];
test_inputs  = [test2.';test3.'];
train_targets  = ones(size(train2,2)+size(train3,2),1 );
valid_targets  = ones(size(valid2,2)+size(valid3,2),1 );
test_targets  = ones(size(test2,2)+size(test3,2),1 );

%% Initiate k values
k_values = [1,3,5,7,9];

%% Plot classification_rate - k_value figure for validation set.
plot_k_cr(k_values, train_inputs, train_targets, valid_inputs, valid_targets);
title('Classification Rate for Validation Set');

%% Plot classification_rate - k_value figure for test set.
figure();
plot_k_cr(k_values, train_inputs, train_targets, test_inputs, test_targets);
title('Classification Rate for Test Set');
