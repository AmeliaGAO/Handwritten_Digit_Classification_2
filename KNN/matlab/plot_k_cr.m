function figure = plot_k_cr(k_values, train_inputs, train_targets, valid_inputs, valid_targets)
% plot_k_cr: 
% Apply knn for every value of k, on train_inputs, train_targets, and
% valid_inputs to get the classification results for the valid set.
% Calculate classficiation rate for each k value, by comparing the classification results
% and the valid_targets. 
% Plot classification rate with k values.
%
% Inputs:
%   k_values: K x 1 matrix for all k values tested.
%   train_inputs: M x D training inputs
%   train_targets: M x 1 training set targets
%   valid_inputs: N x D validation set inputs
%   valid_targets: N x 1 validation set targets
%
% Output:
%   figure: the plot of classification rate and k values. 

n = size(k_values,2);
r_values(1:n,1) = 0;
for i = 1:n
    valid_results = run_knn(k_values(1,i),train_inputs, train_targets,valid_inputs);
    r_values(i,1)= cal_classification_rate(valid_results, valid_targets);
end

display(r_values);
figure = plot(k_values, r_values, 'b-o');
xlabel('k values');
ylabel('classification rate');
end
