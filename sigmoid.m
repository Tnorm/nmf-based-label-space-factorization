function [ out ] = sigmoid( in, param )
out = 1 ./ (1 + exp( - param * in ) );
end