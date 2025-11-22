function test_probs = train_tree(model_dataset,model_info,parameters,iter)

% Make parameters
fwindows_train_block = model_dataset.fwindows_train_block;
fwindows_test_block = model_dataset.fwindows_test_block;
Yi_train_vec = model_dataset.Yi_train_vec;
Yi_test_vec = model_dataset.Yi_test_vec;
paramHash_model = model_info.paramHash_model;
load_path = model_info.load_path;
delete_model = model_info.delete_model;

% Import model settings
num_trees = parameters.num_trees;
randomize_training = parameters.randomize_training;

% Load model
try
    if randomize_training
        error("Randomized training doesn't save models.")
    elseif delete_model
        error("Model must be remade for deleted configurations.")
    else
        % Try to load file
        model_file = load((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))));
        model = model_file.beta;
        test_probs = model_file.test_probs;
        true_labels = model_file.true_labels;

        % Generate next model if its needed
        if size(model,2) < iter
            model{iter} = TreeBagger(num_trees, fwindows_train_block, Yi_train_vec, 'Method', 'classification', 'OOBPrediction', 'On');

            % Test data
            [~, test_probs{iter}] = predict(model{iter}, fwindows_test_block);
            true_labels(:,iter) = Yi_test_vec;

            % Save data file
            save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"model","test_probs","true_labels");
        end
    end
catch
    % Train random forest model
    model{iter} = TreeBagger(num_trees, fwindows_train_block, Yi_train_vec, 'Method', 'classification', 'OOBPrediction', 'On');
    [~, test_probs{iter}] = predict(model{iter}, fwindows_test_block);
    true_labels = Yi_test_vec;

    if ~randomize_training
        % Save data file
        save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"model","test_probs","true_labels");
    end
end