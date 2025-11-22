function test_probs = train_nn_cnn1d(model_dataset,model_info,parameters,iter)
warning("off","parallel:gpu:device:DeviceDeprecated");

% Make parameters
fwindows_train_block = model_dataset.fwindows_train_block;
fwindows_test_block = model_dataset.fwindows_test_block;
Yi_train_vec = model_dataset.Yi_train_vec;
Yi_test_vec = model_dataset.Yi_test_vec;
paramHash_model = model_info.paramHash_model;
load_path = model_info.load_path;
delete_model = model_info.delete_model;

% Import model settings
randomize_training = parameters.randomize_training;
cv_spec = parameters.cv_spec;
max_epochs = parameters.max_epochs;
num_filters = parameters.num_filters;
learning_rate = parameters.learning_rate;

% Hyperparameter conversion
validation_percentage = 1 / cv_spec;

% Input format
T = full(ind2vec(Yi_train_vec'+1));   % One-hot targets, size nClasses x nSamples

% Build network layers
num_features = size(fwindows_train_block,2);
numClasses = size(T,1);
layers = [
    sequenceInputLayer(1,"MinLength",num_features,"Normalization","none")
    convolution1dLayer(5,num_filters)
    batchNormalizationLayer
    reluLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    ];

% Get validation set
num_ones = round(validation_percentage * size(fwindows_train_block,1));
valid_perm = randperm(size(fwindows_train_block,1),num_ones);
valid_idx = false(size(fwindows_train_block,1),1);
valid_idx(valid_perm) = true;
fwindows_valid_block = fwindows_train_block(valid_idx,:);
Yi_valid_vec = Yi_train_vec(valid_idx,:);
fwindows_train_block = fwindows_train_block(~valid_idx,:);
Yi_train_vec = Yi_train_vec(~valid_idx,:);

% Reshape/reformat to get valid size for model
XTrain = mat2cell(fwindows_train_block, ones(size(fwindows_train_block,1),1), num_features);
XValid = mat2cell(fwindows_valid_block, ones(size(fwindows_valid_block,1),1), num_features);
XTest  = mat2cell(fwindows_test_block, ones(size(fwindows_test_block,1),1), num_features);
XTrain = cellfun(@(x) x.', XTrain, 'UniformOutput', false);
XValid = cellfun(@(x) x.', XValid, 'UniformOutput', false);
XTest_array = cat(3, XTest{:});   % [1 x 98 x 1907] → [channels x timesteps x batch]
XTest_dl = dlarray(XTest_array, 'CTB');  % C=1, T=98, B=1907

% Set model options
opts = trainingOptions("adam", ...
    "MaxEpochs", max_epochs, ...
    "InitialLearnRate", learning_rate, ...
    "ExecutionEnvironment","gpu", ...
    "Shuffle","every-epoch", ...
    "ValidationData", {XValid, categorical(Yi_valid_vec)}, ...
    "Verbose", false, ...
    "Plots", "none");

% Load model
try
    if randomize_training
        error("Randomized training doesn't save models.")
    elseif delete_model
        error("Model must be remade for deleted configurations.")
    else
        % Try to load file
        model_file = load((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))));
        model = model_file.model;
        test_probs = model_file.test_probs;
        true_labels = model_file.true_labels;

        % Generate next model if its needed
        if size(model,2) < iter
            % Train network
            model_inst = trainnet(XTrain, categorical(Yi_train_vec), layers, "crossentropy", opts);

            % Predict classes
            Ypred = predict(model_inst, XTest_dl);
            test_probs{iter} = double(extractdata(Ypred)).';

            true_labels(:,iter) = Yi_test_vec;
            model{iter} = model_inst;
            if ~randomize_training
                % Save data file
                save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"model","test_probs","true_labels");
            end
        end
    end
catch

    % Train network
    model_inst = trainnet(XTrain, categorical(Yi_train_vec), layers, "crossentropy", opts);

    % Predict classes
    Ypred = predict(model_inst, XTest_dl);
    test_probs{iter} = double(extractdata(Ypred)).';

    true_labels = Yi_test_vec;
    model{iter} = model_inst;
    if ~randomize_training
        % Save data file
        save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"model","test_probs","true_labels");
    end
end

end