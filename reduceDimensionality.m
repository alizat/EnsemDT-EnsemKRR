function [drugFeatVectors,targetFeatVectors]=reduceDimensionality(drugFeatVectors,targetFeatVectors,Y)
%reduceDimensionality reduces the dimensionality of supplied drug and
%target feature vectors
%
% INPUT:
%  drugFeatVectors:     drug feature vectors
%  targetFeatVectors:   target feature vectors
%  Y:                   interaction matrix (required for PLS)
%
% OUTPUT:
%  drugFeatVectors:     'reduced' drug feature vectors
%  targetFeatVectors:   'reduced' target feature vectors

    global dimReduction

    % determine user-selected dimensionality reduction technique
    switch dimReduction
        case 0, technique = str2func('returnAsIs');
        case 1, technique = str2func('reduceDimensionalitySVD');
        case 2, technique = str2func('reduceDimensionalityPLS');
        case 3, technique = str2func('reduceDimensionalityLAPEIG');

        otherwise
            warning('unknown dimensionality reduction method specified')
            warning('will skip doing dimensionality reduction')
            technique = str2func('returnAsIs');
    end

    % dimensionality reduction
    [drugFeatVectors,targetFeatVectors] = technique(drugFeatVectors, targetFeatVectors, Y);
end

%==========================================================================
%==========================================================================

% No dimensionality reduction
function [drugFeatVectors,targetFeatVectors]=returnAsIs(drugFeatVectors,targetFeatVectors,~)
end

%==========================================================================
%==========================================================================

% SVD: Truncated SVD
function [drugFeatVectors,targetFeatVectors]=reduceDimensionalitySVD(drugFeatVectors,targetFeatVectors,~)
%     drugFeatVectors = zscore(drugFeatVectors);
%     targetFeatVectors = zscore(targetFeatVectors);

    global k

    % optimal setting for feature-based methods
    k = 10;
    [u,s,~] = svds(drugFeatVectors,k);        drugFeatVectors = u*s;
    [u,s,~] = svds(targetFeatVectors,k);    targetFeatVectors = u*s;

%     % optimal setting for similarity-based methods
%     k = 100;
%     [u,s,~] = svds(drugFeatVectors,k);        drugFeatVectors = u*s;
%     [u,s,~] = svds(targetFeatVectors,2*k);    targetFeatVectors = u*s;
end

%==========================================================================
%==========================================================================

% PLS: Partial Least Squares
function [drugFeatVectors,targetFeatVectors]=reduceDimensionalityPLS(drugFeatVectors,targetFeatVectors,Y)
%     drugFeatVectors = zscore(drugFeatVectors);
%     targetFeatVectors = zscore(targetFeatVectors);

    global k
    k = 15;     % feature-based methods
    %k = 25;     % similarity-based methods
    [~,~,drugFeatVectors]   = plsregress(drugFeatVectors,   Y,  k);
    [~,~,targetFeatVectors] = plsregress(targetFeatVectors, Y', k);
end

%==========================================================================
%==========================================================================

% LapEig: Laplacian Eigenmaps
function [drugFeatVectors,targetFeatVectors]=reduceDimensionalityLAPEIG(drugFeatVectors,targetFeatVectors,~)
    global k t
    k = 20;
    t = 7;
    drugFeatVectors = lapeiger(drugFeatVectors, k, t);
    targetFeatVectors = lapeiger(targetFeatVectors, k, t);
end

% ----------------------------------------------------------------------

function Y=lapeiger(X, k, t)
    %X = bsxfun(@minus, X, mean(X));
    W = exp(-squareform(pdist(X)) / size(X,2));
    W = preprocess_pNN(W, t);   % knn graph
    %W = dijkstra(W, 1:size(W,1));       % compute shortest paths
    D = diag(sum(W));
    L = D - W;
    %L = sqrt(D) * L * sqrt(D);

    [U, sigma] = eig(L);
    [~, ind] = sort(real(diag(sigma))); 
    Y = U(:,ind(2:(k+1)));
end

% ----------------------------------------------------------------------

function S=preprocess_pNN(S,p)
%preprocess_PNN sparsifies the similarity matrix S by keeping, for each
%drug/target, the p nearest neighbors and discarding the rest.
%
% S = preprocess_PNN(S,p)

    NN_mat = zeros(size(S));

    % for each drug/target...
    for j=1:length(NN_mat)
        row = S(j,:);                           % get row corresponding to current drug/target
        row(j) = 0;                             % ignore self-similarity
        [~,indx] = sort(row,'descend');         % sort similarities descendingly
        indx = indx(1:p);                       % keep p NNs
        NN_mat(j,indx) = S(j,indx);             % keep similarities to p NNs
        NN_mat(j,j) = S(j,j);                   % also keep the self-similarity (typically 1)
    end

    % symmetrize the modified similarity matrix
    %S = (NN_mat+NN_mat')/2;
    S = max(NN_mat, NN_mat');

end