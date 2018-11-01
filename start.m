clear all

%-------------------------------------------------------------------

warning off
diary off; diary on;
fprintf('\nSTART TIME:    %s\n\n', datestr(now));

%-------------------------------------------------------------------

global drugFeatureVectors targetFeatureVectors

% load data
path = 'data/';
%path = 'data_tabei/';
Y = importdata([path 'interactionMatrix.txt']);
drugFeatureVectors = importdata([path 'drugFeatureVectors.txt']);
targetFeatureVectors = importdata([path 'targetFeatureVectors.txt']);

%-------------------------------------------------------------------

global batchSize gridSearchMode

batchSize = 10000;      % batch size

% which methods to run?
predictionMethods = {'dt','rf','np','wp','nbi','rls','rls_kron','ensemdt','ensemkrr'};    % ,'svm'  (NOTE: SVM takes quite a long time to execute!)

% CV settings to consider?
cv_settings = 1;            % fixed to S1

% dimensionality reduction techniques to consider?
dimRedTechniques = 0:3;     % 0:none, 1:SVD, 2:PLS, 3:LapEig

%-------------------------------------------------------------------

global predictionMethod cv_setting npRatio numLearners r dimReduction

for pm=1:length(predictionMethods)
    disp('==================================================')
    predictionMethod = predictionMethods{pm};
    if ismember(predictionMethod, {'dt', 'svm', 'rf'})
        npRatio = 1;            % -ve to +ve ratio
        
    elseif strcmp(predictionMethod, 'ensemdt')
        npRatio = 5;            % -ve to +ve ratio
        numLearners = 50;       % number of base learners
        r = 0.2;                % percentage of features to be used per base learner (feature subspacing)

    elseif strcmp(predictionMethod, 'ensemkrr')
        numLearners = 50;       % number of base learners
        r = 0.2;                % percentage of features to be used per base learner (feature subspacing)
    end

    % loop over all cross validation settings
    for cvs=cv_settings
        disp('-------------------')

        for dimReduction=dimRedTechniques
            % CV setting
            cv_setting = ['S' int2str(cvs)];
            switch cv_setting
                case 'S1', disp('CV Setting Used: S1 - PAIR');
                case 'S2', disp('CV Setting Used: S2 - DRUG');
                case 'S3', disp('CV Setting Used: S3 - TARGET');
            end
            disp(' ')

            %-----------------------------

            % print parameters
            %disp(['     batchSize = ' num2str(batchSize)])
            %disp(['gridSearchMode = ' num2str(gridSearchMode)])
            %disp(' ')
            disp(['Prediction method = ' predictionMethod])
            if ismember(predictionMethod,{'dt','rf','svm','ensemdt'})
                disp(['          npRatio = ' num2str(npRatio)])
            end
            if ismember(predictionMethod, {'ensemdt', 'ensemkrr'})
                disp(['      numLearners = ' num2str(numLearners)])
                disp(['                r = ' num2str(r)])
            end
            disp(['     dimReduction = ' num2str(dimReduction)])

            %-----------------------------

            % run chosen selection method and output CV results
            gridSearchMode = 0;
            tic
            scores = crossValidation(Y);
            disp(' ')
            toc
            fprintf('\n\nAUC:\t%.3g\n\n\n', scores.auc)

            %-----------------------------

%             % example of grid search mode
%             gridSearchMode = 1;
%             for numLearners=100:100:500
%                 scores = crossValidation(Y);
%                 fprintf('\nnumLearners=%g, AUC: %.3g', numLearners, scores.auc)
%             end
%             disp(' ')
%             disp(' ')

            %-----------------------------

            diary off; diary on;
        end

        disp('-------------------')
        disp(' ')
        disp(' ')
        diary off; diary on;
    end
    
    disp('==================================================')
    diary off; diary on;
end
diary off;

%-------------------------------------------------------------------