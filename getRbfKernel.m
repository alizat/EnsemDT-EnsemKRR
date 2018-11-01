function K = getRbfKernel(X)
    K = exp(-squareform(pdist(X).^2) / size(X, 2));

%     % Alternative 1
%     X = X';
%     n1sq = sum(X.^2,1);
%     n1 = size(X,2);
%     D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*(X'*X);
%     K = exp(-D/size(X,2));

%     % Alternative 2
%     nsq=sum(X.^2,2);
%     K=bsxfun(@minus,nsq,(2*X)*X.');
%     K=bsxfun(@plus,nsq.',K);
%     K=exp(-K/size(X,2));
end