% =========================================================================
%                          Written by Yi Qin and Xin Wang
% =========================================================================
function nn = dbnunfoldtonn(dbn)
%DBNUNFOLDTONN Unfolds a DBN to a NN
%   dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
%   layer of size outputsize added.

    nn = nnsetup(dbn.opts.netsize);
    for i = 1 : numel(dbn.rbm)
        nn.W{i} = [dbn.rbm{i}.c dbn.rbm{i}.W];
    end
end

