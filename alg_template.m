function Yhat=alg_template(Y,predictionMethod,test_indices)
%alg_template predicts DTIs based on the prediction method selected in
%start.m or sensitivity_analysis.m
%
% INPUT:
%  Y:                     interaction matrix
%  predictionMethod:      method to use for prediction
%  test_indices:          indices of the test set instances
%
% OUTPUT:
%  Yhat:                  prediction scores matrix

    % Parameters
    global drugFeatureVectors targetFeatureVectors

    switch predictionMethod
        % proposed ensemble framework --------------------
        case 'ensemdt'
            Yhat = alg_ensemble(Y, test_indices, 'dt');
        case 'ensemkrr'
            Yhat = alg_ensemble(Y, test_indices, 'rls');

        % similarity-based methods -----------------------
        case {'rls', 'rls_kron', 'np', 'wp', 'nbi'}
            predFn = str2func(['alg_'  predictionMethod]);
            Yhat = predFn(Y);

        % feature-based methods --------------------------
        otherwise
            % dimensionality reduction
            [drugFeatVectors,targetFeatVectors] = reduceDimensionality(drugFeatureVectors,targetFeatureVectors,Y);

            % generate training set
            [patterns, labels, ~] = generateTrainingSet(Y, test_indices, drugFeatVectors, targetFeatVectors);

            % train predictive model
            switch predictionMethod
                case 'dt',      predModel.model = compact(fitctree(patterns, labels, 'MinLeafSize', 10));
                case 'rf',      predModel.model = compact(TreeBagger(500, patterns, labels, 'Prior', 'uniform'));
                case 'svm',     predModel.model = compact(fitcsvm(patterns, labels, 'KernelFunction', 'rbf'));
            end
            clear patterns labels

            % predict
            Yhat = predictor(predModel, test_indices, predictionMethod, drugFeatVectors, targetFeatVectors);
    end

end


% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %


function Yhat=alg_ensemble(Y,test_indices,baseLearner)
%alg_ensemble predicts DTIs based on the algorithm described in the
%following paper (but without the dimensionality reduction):
% Ali Ezzat, Peilin Zhao, Min Wu, Xiao-Li Li and Chee-Keong Kwoh
% (2017) Drug-Target Interaction Prediction using Ensemble Learning and
%           Dimensionality Reduction 
%
% INPUT:
%  Y:                     interaction matrix
%  test_indices:          indices of the test set instances
%  baseLearner:           base learner to be used in the ensemble
%
% OUTPUT:
%  Yhat:                  prediction matrix (including test set scores)

    % Parameters
    global numLearners drugFeatureVectors targetFeatureVectors r

    % instances to be excluded from training set
    exclude_indices = test_indices;

    % generate models
    Yhat = 0;
    for c = 1:numLearners
        % feature subspacing
        numDrugFeatures = size(drugFeatureVectors, 2);
        numTargetFeatures = size(targetFeatureVectors, 2);
        drugFeatures = randperm(numDrugFeatures, floor(numDrugFeatures*r));
        targetFeatures = randperm(numTargetFeatures, floor(numTargetFeatures*r));
        drugFeatVectors = drugFeatureVectors(:, drugFeatures);
        targetFeatVectors = targetFeatureVectors(:, targetFeatures);

        % dimensionality reduction
        [drugFeatVectors,targetFeatVectors] = reduceDimensionality(drugFeatVectors, targetFeatVectors, Y);

        if ismember(baseLearner, {'rls', 'rls_kron', 'np', 'wp', 'nbi'})
            predFn = str2func(['alg_'  baseLearner]);
            Yhat = Yhat + predFn(Y, drugFeatVectors, targetFeatVectors);

        else
            % train
            [patterns, labels, ~] = generateTrainingSet(Y, exclude_indices, drugFeatVectors, targetFeatVectors);
            predModel.model = compact(fitctree(patterns, labels, 'Prior', 'uniform'));
            Yhat = Yhat + predictor(predModel, test_indices, baseLearner, drugFeatVectors, targetFeatVectors);
        end
    end
    clear patterns labels pos_indices drugFeatVectors targetFeatVectors
end


% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %


function Yhat=predictor(predModel,test_indices,algorithm,drugFeatVectors,targetFeatVectors)
%predictor is a helper function that produces prediction scores for
%feature-based DTI prediction methods

    global batchSize

    % to hold prediction scores
    predScores = zeros(length(test_indices), 1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % NOTE:                                                            %
    % the entire set of feature vectors of the test set can't fit into %
    % memory, so we do predictions of test set instances in batches    %
    % (default batch size = 10000)                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % collect predictions
    for counter = 0:batchSize:length(test_indices)
        % current batch of test set instances
        start_index = counter + 1;
        end_index = min(counter + batchSize, length(test_indices));
        if start_index > length(test_indices), break; end

        % prepare testing instances
        testingFeatureVectors = generateFeatures(test_indices(start_index:end_index), drugFeatVectors, targetFeatVectors);

        % get prediction scores
        predScores(start_index:end_index) = predictWithModel(algorithm, predModel, testingFeatureVectors);

        clear testingFeatureVectors
    end

    Yhat = zeros(size(drugFeatVectors, 1), size(targetFeatVectors, 1));
    Yhat(test_indices) = predScores;
end

% ----------------------------------------------------------------------

function scores=predictWithModel(algorithm,predModel,X)

    switch algorithm
        % DECISION TREE, RANDOM FOREST, SVM ----------
        case {'dt', 'rf', 'svm'}
            % prediction scores (probabilities)
            [~, scores] = predict(predModel.model, X);
            scores = scores(:,2);

        % OUR METHOD ---------------------------------
        otherwise
            % predicted labels
            scores = predict(predModel.model, X);
            if iscell(scores), scores = str2double(scores); end
    end
end


% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %


function Yhat=alg_rls(Y,drugFeatVectors,targetFeatVectors)
%alg_rls_kron predicts DTIs based on the algorithm described in the following paper: 
% Twan van Laarhoven, Elena Marchiori,
% (2011) Gaussian interaction profile kernels for predicting drug–target
%           interaction 
%
% http://cs.ru.nl/~tvanlaarhoven/drugtarget2011/

    % Parameters
    global drugFeatureVectors targetFeatureVectors

    if nargin == 1
        % dimensionality reduction
        [drugFeatVectors,targetFeatVectors] = reduceDimensionality(drugFeatureVectors,targetFeatureVectors,Y);
    end

    % generate similarity matrices
    ka = getRbfKernel(drugFeatVectors);
    kb = getRbfKernel(targetFeatVectors);

    % WNN
    eta = 0.7;
    Y = preprocess_WNN(Y,ka,kb,eta);

    % GIP
    alpha = 0.5;
    ka = alpha*ka + (1-alpha)*getGipKernel(Y);
    kb = alpha*kb + (1-alpha)*getGipKernel(Y');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Regularized Least Squares (RLS-avg) %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%     % Subtract the mean value (optional)
%     mean_value = full(mean(Y(:)));
%     Y = Y - mean_value;
	
	% Predict values with GP method
	sigma = 1;
	[na,nb] = size(Y);
	y2a = ka * ((ka+sigma*eye(na)) \ Y);
	y2b = (Y / (kb+sigma*eye(nb))) * kb;
	Yhat = (y2a+y2b)/2;

%     % Add subtracted mean value from before
%     Yhat = Yhat + mean_value;
end


% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %


function Yhat=alg_rls_kron(Y,drugFeatVectors,targetFeatVectors)
%alg_rls_kron predicts DTIs based on the algorithm described in the following paper: 
% Twan van Laarhoven, Elena Marchiori,
% (2013) Predicting drug–target interactions for new drug compounds using a
%           weighted nearest neighbor profile 
%
% http://cs.ru.nl/~tvanlaarhoven/drugtarget2013/

    % Parameters
    global drugFeatureVectors targetFeatureVectors

    if nargin == 1
        % dimensionality reduction
        [drugFeatVectors,targetFeatVectors] = reduceDimensionality(drugFeatureVectors,targetFeatureVectors,Y);
    end

    % generate similarity matrices
    ka = getRbfKernel(drugFeatVectors);
    kb = getRbfKernel(targetFeatVectors);

    % WNN
    eta = 0.7;
    Y = preprocess_WNN(Y,ka,kb,eta);

    % GIP
    alpha = 0.5;
    ka = alpha*ka + (1-alpha)*getGipKernel(Y);
    kb = alpha*kb + (1-alpha)*getGipKernel(Y');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Regularized Least Squares (RLS-kron) %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % RLS_KRON
	sigma = 1;
	[va,la] = eig(ka);
	[vb,lb] = eig(kb);
	l = kron(diag(lb)',diag(la));
	l = l ./ (l + sigma);
	m1 = va' * Y * vb;
	m2 = m1 .* l;
	Yhat = va * m2 * vb';
end


% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %


function Yhat=alg_np(Y,drugFeatVectors,targetFeatVectors)
%alg_np predicts DTIs based on the Nearest Profile algorithm described in the following paper: 
% Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda and Minoru Kanehisa,
% (2008) Prediction of drug–target interaction networks from the integration of chemical and genomic spaces

    % Parameters
    global drugFeatureVectors targetFeatureVectors

    % if drug and target feature vectors are not supplied...
    if nargin == 1
        % dimensionality reduction on full feature sets
        [drugFeatVectors,targetFeatVectors] = reduceDimensionality(drugFeatureVectors,targetFeatureVectors,Y);
    end

    % generate similarity matrices
    Sd = getRbfKernel(drugFeatVectors);
    St = getRbfKernel(targetFeatVectors);

%     % WNN
%     eta = 0.7;
%     Y = preprocess_WNN(Y,Sd,St,eta);
% 
%     % GIP
%     alpha = 0.5;
%     Sd = alpha*Sd + (1-alpha)*getGipKernel(Y);
%     St = alpha*St + (1-alpha)*getGipKernel(Y');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Nearest Profile (NP) %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Nearest Profile
    Sd(logical(eye(length(Sd)))) = 0;   % remove self-similarities
    [maxx, indx] = max(Sd);             % get nearest neighbor for each drug
    for i=1:length(Sd)
        Sd(i, :) = 0;                   % ignore all, ...
        Sd(i, indx(i)) = maxx(i);       % but the nearest neighbor
    end
    St(logical(eye(length(St)))) = 0;   % remove self-similarities
    [maxx, indx] = max(St);             % get nearest neighbor for each target
    for j=1:length(St)
        St(j, :) = 0;                   % ignore all, ...
        St(j, indx(j)) = maxx(j);       % but the nearest neighbor
    end
    yd = Sd * Y;
    yt = (St * Y')';
    Yhat = (yd + yt) / 2;
end


% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %


function Yhat=alg_wp(Y,drugFeatVectors,targetFeatVectors)
%alg_wp predicts DTIs based on the Weighted Profile algorithm described in the following paper: 
% Yoshihiro Yamanishi, Michihiro Araki, Alex Gutteridge, Wataru Honda and Minoru Kanehisa,
% (2008) Prediction of drug–target interaction networks from the integration of chemical and genomic spaces

    % Parameters
    global drugFeatureVectors targetFeatureVectors

    % if drug and target feature vectors are not supplied...
    if nargin == 1
        % dimensionality reduction on full feature sets
        [drugFeatVectors,targetFeatVectors] = reduceDimensionality(drugFeatureVectors,targetFeatureVectors,Y);
    end

    % generate similarity matrices
    Sd = getRbfKernel(drugFeatVectors);
    St = getRbfKernel(targetFeatVectors);

%     % WNN
%     eta = 0.7;
%     Y = preprocess_WNN(Y,Sd,St,eta);
% 
%     % GIP
%     alpha = 0.5;
%     Sd = alpha*Sd + (1-alpha)*getGipKernel(Y);
%     St = alpha*St + (1-alpha)*getGipKernel(Y');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Weighted Profile (WP) %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Weighted Profile
    yd = bsxfun(@rdivide, Sd * Y, sum(Sd,2));   yd(Y==1) = 1;
    yt = bsxfun(@rdivide, Y * St, sum(St));     yt(Y==1) = 1;
    Yhat = (yd + yt) / 2;
end


% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %
% *********************************************************************** %


function Yhat=alg_nbi(Y,drugFeatVectors,targetFeatVectors)
%alg_nbi predicts DTIs based on the algorithm described in the following paper: 
% Feixiong Cheng, Chuang Liu, Jing Jiang, Weiqiang Lu, Weihua Li, Guixia Liu, Weixing Zhou, Jin Huang, Yun Tang
% (2012) Prediction of Drug-Target Interactions and Drug Repositioning via Network-Based Inference

    % Parameters
    global drugFeatureVectors targetFeatureVectors

    % if drug and target feature vectors are not supplied...
    if nargin == 1
        % dimensionality reduction on full feature sets
        [drugFeatVectors,targetFeatVectors] = reduceDimensionality(drugFeatureVectors,targetFeatureVectors,Y);
    end

    % generate similarity matrices
    Sd = getRbfKernel(drugFeatVectors);
    St = getRbfKernel(targetFeatVectors);

%     % WNN
%     eta = 0.7;
%     Y = preprocess_WNN(Y,Sd,St,eta);
% 
%     % GIP
%     alpha = 0.5;
%     Sd = alpha*Sd + (1-alpha)*getGipKernel(Y);
%     St = alpha*St + (1-alpha)*getGipKernel(Y');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Network-based Inference (NBI) %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % normalize Sd and St
    Sd = Sd ./ (sum(Sd,2) * sum(Sd));
    St = St ./ (sum(St,2) * sum(St));
    % normalization idea borrowed from the following publication
    % Wenhui Wang, Sen Yang, Jing Li
    % (2013) Drug target predictions based on heterogeneous graph inference

    % NBI
    Yhat = Y;
    alpha = 0.5;
    Yhat = (alpha * Sd * Yhat * St) + ((1 - alpha) * Y);
end