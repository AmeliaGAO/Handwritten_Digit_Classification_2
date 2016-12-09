function r = cal_classification_rate(results, targets)
%    Compute classfication rate.
%    Inputs:
%        targets : N x 1 vector of targetes.
%        results : N x 1 vector of classification result.
%    Outputs:
%        r : number of correctly predicted cases, divided by total number
%        of data points.

error(nargchk(2,2,nargin));

if (size(results,1) ~= size(targets,1))
   error('Results and Targets should be of same dimensionality');
end

N = size(results,1);
n = 0;
for i = 1:size(results,1)
   if results(i,1) == targets(i,1)
       n = n+1;
   end
end

r = n*1.0/N;
end