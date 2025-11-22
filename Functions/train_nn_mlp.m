function test_probs = train_nn_mlp(model_dataset,model_info,parameters,iter)

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
max_fail = parameters.max_fail;
min_grad = parameters.min_grad;
hidden_neurons = parameters.hidden_neurons;
backprop_func = parameters.backprop_func;
L2_regularization = parameters.L2_regularization;
max_epochs = parameters.max_epochs;

% Hyperparameter conversion
validation_percentage = 1 / cv_spec;

% Input format
T = full(ind2vec(Yi_train_vec'+1));   % One-hot targets, size nClasses x nSamples

% Build network
model_inst = patternnet(hidden_neurons, backprop_func);
model_inst.inputs{1}.processFcns = {}; % Disable internal normalization

% Data splitting for validation
model_inst.divideFcn = 'dividerand';   % random split for validation
model_inst.divideParam.trainRatio = 1 - validation_percentage;
model_inst.divideParam.valRatio   = validation_percentage;
model_inst.divideParam.testRatio  = 0;

% Early stopping and regularization
model_inst.performParam.regularization = L2_regularization;
model_inst.trainParam.max_fail         = max_fail;
model_inst.trainParam.epochs           = max_epochs;
model_inst.trainParam.min_grad         = min_grad;

% Restructure data
if strcmp(string(backprop_func),"traingd") || strcmp(string(backprop_func),"traingda") || strcmp(string(backprop_func),"traingdm")
    x = gpuArray(fwindows_train_block');
    t = gpuArray(T);
    model_inst.trainParam.useGPU = 'yes';
else
    x = fwindows_train_block';
    t = T;
end
model_inst.trainParam.showWindow = false;
model_inst.trainParam.showCommandLine = false;
model_inst.trainParam.show = NaN;

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
        train_record = model_file.train_record;

        % Generate next model if its needed
        if size(model,2) < iter
            [model_inst, train_record{iter}] = train(model_inst, fwindows_train_block', T);

            % Predict classes
            Ypred = model_inst(fwindows_test_block');
            test_probs{iter} = Ypred';

            true_labels(:,iter) = Yi_test_vec;
            model{iter} = model_inst;
            if ~randomize_training
                % Save data file
                save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"model","test_probs","true_labels",'train_record');
            end
        end
    end
catch

    [model_inst, train_record{iter}] = train(model_inst, fwindows_train_block', T);

    % Predict classes
    Ypred = model_inst(fwindows_test_block');
    test_probs{iter} = Ypred';

    true_labels = Yi_test_vec;
    model{iter} = model_inst;
    if ~randomize_training
        % Save data file
        save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"model","test_probs","true_labels",'train_record');
    end
end


