function [kernelMat, sigm] = calc_kernel(Xtrain)

nsq = sum(Xtrain .^ 2, 2);
dist = bsxfun(@minus, nsq, (2 * Xtrain) * Xtrain.');
dist = bsxfun(@plus, nsq.', dist);

dist_sigm = dist .^ (1/2);
sigm = 2 * mean(dist_sigm(:));
kernelMat = gauss_kernel(dist, sigm);

end

function [ out ] = gauss_kernel(dist, sigm)
out = exp(- 1 / (2 * sigm^2) * dist);
end

