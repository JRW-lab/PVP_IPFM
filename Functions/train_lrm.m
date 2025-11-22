function test_probs = train_lrm(model_dataset,model_info,parameters,iter)

% Make parameters
fwindows_train_block = model_dataset.fwindows_train_block;
fwindows_test_block = model_dataset.fwindows_test_block;
Yi_train_vec = model_dataset.Yi_train_vec;
Yi_test_vec = model_dataset.Yi_test_vec;
paramHash_model = model_info.paramHash_model;
load_path = model_info.load_path;
delete_model = model_info.delete_model;

% Import model settings
alpha = parameters.alpha;
randomize_training = parameters.randomize_training;
cv_spec = parameters.cv_spec;
max_iterations = parameters.max_iterations;

% Select logistic regression or ordinal regression
if length(unique(Yi_train_vec)) > 2
    lrm_type = "ordinal";
else
    lrm_type = "elastic";
end

% Load model
try
    if randomize_training
        error("Randomized training doesn't save models.")
    elseif delete_model
        error("Model must be remade for deleted configurations.")
    else
        % Try to load file
        models = load((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))));
        beta = models.beta;
        theta = models.theta;
        test_probs = models.test_probs;
        true_labels = models.true_labels;

        % Temporary conversions - remove after all old data is gone
        if ~iscell(test_probs)
            test_probs = {test_probs};
            save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"beta","theta","test_probs","true_labels");
        end
        prob_size = unique(cellfun(@(x) size(x,2),test_probs));
        if prob_size == 1
            for i = 1:length(test_probs)
                test_probs_sel = test_probs{i};
                test_probs{i} = [(1-test_probs_sel) test_probs_sel];
            end
            save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"beta","theta","test_probs","true_labels");
        end

        % Generate next model if its needed
        if size(models.beta,2) < iter
            switch lrm_type
                case "elastic"
                    % Find best fit for data using training data
                    [beta_lasso,fit] = lassoglm(fwindows_train_block,Yi_train_vec,'binomial','NumLambda',10,'CV',cv_spec,'Alpha',alpha,'MaxIter',max_iterations);

                    % Add first element of beta
                    beta_0 = fit.Intercept;
                    indx = fit.IndexMinDeviance;
                    beta(:,iter) = [beta_0(indx);beta_lasso(:,indx)];
                    theta(1,iter) = NaN;
                case "ordinal"
                    % Train model
                    model = fitOrdinalRegression(fwindows_train_block,Yi_train_vec+1,length(unique(Yi_train_vec)));
                    beta(:,iter) = model.beta;
                    theta(:,iter) = model.theta;
            end

            % Test data
            test_probs{iter} = regression_test(lrm_type,fwindows_test_block,beta(:,iter),theta(:,iter));
            true_labels(:,iter) = Yi_test_vec;

            % Save data file
            save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"beta","theta","test_probs","true_labels");
            % beta = beta(:,iter);
        end
    end
catch
    switch lrm_type
        case "elastic"
            % Find best fit for data using training data
            [beta_lasso,fit] = lassoglm(fwindows_train_block,Yi_train_vec,'binomial','NumLambda',10,'CV',cv_spec,'Alpha',alpha,'MaxIter',max_iterations);

            % Add first element of beta
            beta_0 = fit.Intercept;
            indx = fit.IndexMinDeviance;
            beta = [beta_0(indx);beta_lasso(:,indx)];
            theta = NaN;
        case "ordinal"
            % Train model
            model = ordinalglm(fwindows_train_block,Yi_train_vec+1,length(unique(Yi_train_vec)));
            beta = model.beta;
            theta = model.theta;
    end

    % Test data
    test_probs{iter} = regression_test(lrm_type,fwindows_test_block,beta,theta);
    true_labels = Yi_test_vec;

    if ~randomize_training
        % Save data file
        save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"beta","theta","test_probs","true_labels");
    end
end