function model_fun_v3(save_data,conn,table_name,parameters,paramHash,iter,delete_model)

% Settings
load_path = "Models/";

% If the folder does not exist, create it
if ~exist(load_path, 'dir')
    mkdir(load_path);
end

% Load data from DB and set new frame count
switch save_data.priority
    case "mysql"
        if save_data.save_mysql
            try
                T = mysql_load(conn,table_name,"*");
            catch
                conn = mysql_login(conn.DataSource);
                T = mysql_load(conn,table_name,"*");
            end
        elseif save_data.save_excel
            try
                T = readtable(save_data.excel_path, 'TextType', 'string');
            catch
                T = table;
            end
        end
    case "local"
        if save_data.save_excel
            try
                T = readtable(save_data.excel_path, 'TextType', 'string');
            catch
                T = table;
            end
        elseif save_data.save_mysql
            try
                T = mysql_load(conn,table_name,"*");
            catch
                conn = mysql_login(conn.DataSource);
                T = mysql_load(conn,table_name,"*");
            end
        end
end
try
    sim_result = T(string(T.param_hash) == paramHash, :);
catch
    sim_result = [];
end

if ~isempty(sim_result)
    % Find new frame count to simulate
    if sim_result.frames_simulated < iter
        run_flag = true;
    else
        run_flag = false;
    end
else
    run_flag = true;
end

% Run if new frames are needed
if run_flag

    % Make parameters
    try
        training_percentage = parameters.training_percentage;
    catch
        training_percentage = 0.7;
    end
    try
        cv_spec = parameters.cv_spec;
    catch
        cv_spec = 5;
    end
    try
        max_iterations = parameters.max_iterations;
    catch
        max_iterations = 1e8;
    end
    window_duration = parameters.window_duration;
    frequency_limit = parameters.frequency_limit;
    alpha = parameters.alpha;
    training_type = parameters.training_type;
    split_by = parameters.split_by;
    tshift = parameters.tshift;
    drop_DC_component = parameters.drop_DC_component;
    normalize_power = parameters.normalize_power;
    try
        randomize_training = parameters.randomize_training;
    catch
        randomize_training = false;
    end
    try
        pca_method = parameters.pca_method;
        pca_sigma_threshold = parameters.pca_sigma_threshold;
    catch
        pca_method = "none";
    end

    % Load dataset
    [data,sample_rate] = load_dataset(parameters);
    data_master = vertcat(data.signals{:});

    % Set parameters
    if training_type == "patient"
        single_patient_testing = true;
        parameters = rmfield(parameters,"randomize_training");
        patients_tested = length(data_master);
    elseif training_type == "percentage"
        single_patient_testing = false;
        patients_tested = 1;
    else
        error("Unsupported testing type.")
    end

    % Category setup
    len_signals = cellfun(@numel, data.signals);
    Yi = zeros(sum(len_signals), 1);
    idx = 1;
    for j = 1:length(len_signals)
        Yi(idx:idx+len_signals(j)-1) = j;
        idx = idx + len_signals(j);
    end
    [~, ~, Yi] = unique(Yi, 'stable');
    classes = unique(Yi);
    Yi = Yi - 1;
    if length(unique(Yi)) > 2
        model_type = "ordinal";
    else
        model_type = "elastic";
    end

    % fprintf("Testing model...\n")
    win_spec = zeros(patients_tested,length(classes));
    win_sens = zeros(patients_tested,length(classes));
    win_accy = zeros(patients_tested,1);
    win_conmat = zeros(length(classes),length(classes),patients_tested);
    pat_spec = zeros(1,length(classes));
    pat_sens = zeros(1,length(classes));
    pat_accy = zeros(patients_tested,1);
    pat_conmat = zeros(length(classes));
    for patient_sel = 1:patients_tested

        % Generate t-windows for all desired signals and data types
        if single_patient_testing
            switch split_by
                case "signal"
                    % Select data
                    waveforms_train = data_master([1:patient_sel-1, patient_sel+1:end]);
                    waveforms_test = data_master(patient_sel);
                    Yi_train = Yi([1:patient_sel-1, patient_sel+1:end]);
                    Yi_test = Yi(patient_sel);

                    % Create t-windows
                    twindows_train = cellfun(@(x) make_twindows(x,sample_rate,window_duration,tshift*sample_rate),waveforms_train,"UniformOutput",false);
                    twindows_test = cellfun(@(x) make_twindows(x,sample_rate,window_duration,tshift*sample_rate),waveforms_test,"UniformOutput",false);

                    % Create Yi for training and testing
                    Yi_train_vecs = cellfun(@(x,y) ones(size(x,1),1) * y,twindows_train,num2cell(Yi_train),"UniformOutput",false);
                    Yi_test_vecs = cellfun(@(x,y) ones(size(x,1),1) * y,twindows_test,num2cell(Yi_test),"UniformOutput",false);
                otherwise
                    error("Unsupported split method! LOOCV only supports signal-split training")
            end
        else
            switch split_by
                case "patient"

                    % Use ~training_percentage of patients for training
                    training_sigs = [];
                    testing_sigs = [];
                    Yi_training = [];
                    Yi_testing = [];
                    for k = classes.'
                        patient_locs = Yi == (k-1);
                        patient_sigs = data_master(patient_locs);
                        detection_range = 1:length(patient_sigs);
                        if randomize_training
                            training_locs = randperm(length(detection_range), round(length(detection_range) * training_percentage));
                        else
                            training_locs = 1:round(length(patient_sigs)*training_percentage);
                        end
                        testing_locs = detection_range(~ismember(detection_range,training_locs));
                        training_sigs = [training_sigs; patient_sigs(training_locs)];
                        testing_sigs = [testing_sigs; patient_sigs(testing_locs)];
                        Yi_training = [Yi_training; (k-1)*ones(length(training_locs),1)];
                        Yi_testing = [Yi_testing; (k-1)*ones(length(testing_locs),1)];
                    end

                case "signal"
                    % Use ~training_percentage of each signal for training
                    % Find ranges for training and testing
                    trange_train = cellfun(@(x) 1:floor(training_percentage*length(x)), data_master,"UniformOutput",false);
                    trange_test = cellfun(@(x) ceil(training_percentage*length(x)):length(x), data_master,"UniformOutput",false);

                    % Get signals for training and testing
                    training_sigs = cellfun(@(x,y) x(y), data_master,trange_train,"UniformOutput",false);
                    testing_sigs = cellfun(@(x,y) x(y), data_master,trange_test,"UniformOutput",false);

                    % Copy original Yi locations for training and testing windows
                    Yi_training = Yi;
                    Yi_testing = Yi;
                otherwise
                    error("Unsupported split method!")
            end

            % Create t-windows
            twindows_train = cellfun(@(x) make_twindows(x,sample_rate,window_duration,tshift*sample_rate),training_sigs,"UniformOutput",false);
            twindows_test = cellfun(@(x) make_twindows(x,sample_rate,window_duration,tshift*sample_rate),testing_sigs,"UniformOutput",false);

            % Create Yi for training and testing
            Yi_train_vecs = cellfun(@(x,y) ones(size(x,1),1) * y,twindows_train,num2cell(Yi_training),"UniformOutput",false);
            Yi_test_vecs = cellfun(@(x,y) ones(size(x,1),1) * y,twindows_test,num2cell(Yi_testing),"UniformOutput",false);
        end

        % Create f-windows
        fwindows_train = cellfun(@(x) fft_rhys(x,sample_rate,frequency_limit,window_duration),twindows_train,'UniformOutput',false);
        fwindows_test = cellfun(@(x) fft_rhys(x,sample_rate,frequency_limit,window_duration),twindows_test,'UniformOutput',false);

        % Create locations for test patient data
        test_lengths = cellfun(@(x) size(x,1),fwindows_test,"UniformOutput",false);
        test_locations = cellfun(@(x,y) y.*ones(x,1),test_lengths,num2cell(1:length(test_lengths)).',"UniformOutput",false);

        % Compress cell arrays into workable data blocks
        test_locations_block = vertcat(test_locations{:});
        fwindows_train_block = vertcat(fwindows_train{:});
        fwindows_test_block = vertcat(fwindows_test{:});
        Yi_train_vec = vertcat(Yi_train_vecs{:});
        Yi_test_vec = vertcat(Yi_test_vecs{:});

        % Drop DC component if specified
        if drop_DC_component
            fwindows_train_block = fwindows_train_block(:,2:end);
            fwindows_test_block = fwindows_test_block(:,2:end);
        end

        % Normalize power if specified
        if normalize_power
            row_powers = sum(abs(fwindows_train_block).^2,2);
            row_powers(row_powers == 0) = 1;
            fwindows_train_block = fwindows_train_block ./ sqrt(row_powers);
            row_powers = sum(abs(fwindows_test_block).^2,2);
            row_powers(row_powers == 0) = 1;
            fwindows_test_block = fwindows_test_block ./ sqrt(row_powers);
        end

        % Remove smaller singular values from the data
        if pca_method ~= "none"
            % Normalize training data
            mu = mean(fwindows_train_block);
            sigma = std(fwindows_train_block);
            X_train = (fwindows_train_block - mu) ./ sigma;
            X_testing = (fwindows_test_block - mu) ./ sigma;

            switch pca_method
                case "cov"

                    % Get covariance of normalized data and eigen-decomposition
                    C = cov(X_train);
                    [V,D] = eig(C);

                    % Sort eigenvalues
                    [eigvals, idx] = sort(diag(D), 'descend');
                    V = V(:,idx);

                    % Calculate explained variance ratio
                    evr = eigvals / sum(eigvals);

                case "svd"

                    % Perform SVD
                    [~,S,V] = svd(X_train);

                    % Calculate explained variance ratio
                    s_vals = diag(S);
                    e_var = s_vals.^2 / (size(X_train,1) - 1);
                    evr = e_var / sum(e_var);

                case "kpca"

                    % Find kernel sigma value
                    sq_dists = pdist2(X_train, X_train, 'euclidean').^2;
                    median_dist = median(sq_dists(:));
                    pca_sigma_kernel = sqrt(median_dist / 2);

                    % Generate kernels
                    K_train = exp(-pdist2(X_train, X_train).^2 / (2*pca_sigma_kernel^2));
                    K_test = exp(-pdist2(X_testing,X_train).^2 / (2*pca_sigma_kernel^2));

                    % Center kernels
                    ntr = size(K_train,1);
                    ntt = size(K_test,1);
                    one_tr = ones(ntr,ntr) / ntr;
                    one_tt = ones(ntt,ntr) / ntr;
                    K_train_centered = K_train ...
                        - one_tr * K_train ...
                        - K_train * one_tr...
                        + one_tr * K_train * one_tr;
                    K_test_centered = K_test ...
                        - one_tt*K_train ...
                        - K_test*one_tr ...
                        + one_tt*K_train*one_tr;

                    % Center and decompose kernel matrix
                    [V,D] = eig(K_train_centered);

                    % Sort eigenvalue decomposition
                    eigvals = diag(D);
                    [eigvals, idx] = sort(eigvals, 'descend');
                    V = V(:,idx);

                    % Calculate explained variance ratio
                    evr = eigvals / sum(eigvals);

                case "spca"
                    % Implement sparse PCA
                case "rpca"
                    % Implement robust PCA
                case "ppca"
                    % Probabilistic PCA (MATLAB's ppca?)
                otherwise
                    error("Unexpected PCA method specified!")
            end

            % Find number of desired eigenvalues
            cum_evr = cumsum(evr);
            K = find(cum_evr >= pca_sigma_threshold, 1);

            % Generate reduced datasets, reassign training and testing blocks
            if pca_method == "kpca"
                alphas = V(:,1:K) ./ sqrt(eigvals(1:K))';
                fwindows_train_block = K_train_centered * alphas;
                fwindows_test_block = K_test_centered * alphas;
            else
                fwindows_train_block = X_train * V(:, 1:K);
                fwindows_test_block = X_testing * V(:, 1:K);
            end

        end

        % Serialize to JSON for DB
        parameters_model = parameters;
        if single_patient_testing
            parameters_model.patient_sel = patient_sel;
        end
        paramsJSON_model  = jsonencode_sorted(parameters_model);
        paramHash_model = string(DataHash(paramsJSON_model,'SHA-256'));

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

                % Temporary conversion - remove after all old data is gone
                if ~iscell(test_probs)
                    test_probs = {test_probs};
                    save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"beta","theta","test_probs","true_labels");
                end

                if size(models.beta,2) < iter
                    switch model_type
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
                            model = fitOrdinalRegression(fwindows_train_block,Yi_train_vec+1,length(unique(Yi)));
                            beta(:,iter) = model.beta;
                            theta(:,iter) = model.theta;
                    end

                    % Test data
                    test_probs{iter} = regression_test(model_type,fwindows_test_block,beta,theta);
                    true_labels(:,iter) = Yi_test_vec;

                    % Save data file
                    save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"beta","theta","test_probs","true_labels");
                    beta = beta(:,iter);
                % else
                %     beta = models.beta(:,iter);
                %     theta = theta(:,iter);
                end
            end
        catch
            switch model_type
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
                    model = ordinalglm(fwindows_train_block,Yi_train_vec+1,length(unique(Yi)));
                    beta = model.beta;
                    theta = model.theta;
            end

            % Test data
            test_probs{iter} = regression_test(model_type,fwindows_test_block,beta,theta);
            true_labels = Yi_test_vec;

            if ~randomize_training
                % Save data file
                save((fullfile(load_path, sprintf("model_%s.mat",paramHash_model))),"beta","theta","test_probs","true_labels");
            end
        end

        % Get window-level probabilities
        switch model_type
            case "elastic"
                Yhat_test_vec = (test_probs{iter} > 0.5) * 1;
            case "ordinal"
                [~,Yhat_test_vec] = max(test_probs{iter},[],2);
                Yhat_test_vec = Yhat_test_vec - 1;
        end

        % Get window-level measurements measurements
        win_conmat_inst = confusionmat(Yi_test_vec,Yhat_test_vec, 'Order', classes-1);
        win_conmat(:,:,patient_sel) = win_conmat_inst;
        win_accy(patient_sel) = sum(diag(win_conmat_inst)) ./ sum(win_conmat_inst(:));
        win_sens(patient_sel,:) = diag(win_conmat_inst).' ./ (sum(win_conmat_inst, 2) + eps).';
        for j = 1:length(classes)
            TP = win_conmat_inst(j,j);
            FN = sum(win_conmat_inst(j,:)) - TP;
            FP = sum(win_conmat_inst(:,j)) - TP;
            TN = sum(win_conmat_inst(:)) - TP - FN - FP;
            win_spec(patient_sel,j) = TN / (TN + FP);
        end

        % Get patient-level measurements
        if single_patient_testing
            switch model_type
                case "elastic"
                    patient_yhat = mean(Yhat_test_vec);
                    Yi_hat = patient_yhat > 0.5;
                case "ordinal"
                    Yi_hat = mode(Yhat_test_vec);
            end
            pat_accy(patient_sel) = Yi_hat == Yi(patient_sel);
            % pat_sens(patient_sel,:) = nan(1,length(classes));
            % pat_spec(patient_sel,:) = nan(1,length(classes));
        else
            Yi_hat = zeros(length(data_master),1);
            for k = 1:length(data_master)
                patient_indices = test_locations_block == k;

                if isempty(patient_indices)
                    continue;
                end

                switch model_type
                    case "elastic"
                        patient_yhat = mean(Yhat_test_vec(patient_indices));
                        Yi_hat(k) = patient_yhat > 0.5;
                    case "ordinal"
                        Yi_hat(k) = mode(Yhat_test_vec(patient_indices));
                end
            end

            % Get accuracy measurements
            pat_conmat_inst = confusionmat(Yi, Yi_hat, 'Order', classes-1);
            pat_conmat = pat_conmat_inst;
            pat_accy = sum(diag(pat_conmat_inst)) ./ sum(pat_conmat_inst(:));
            pat_sens(1,:) = diag(pat_conmat_inst).' ./ (sum(pat_conmat_inst, 2) + eps).';
            for j = 1:length(classes)
                TP = pat_conmat_inst(j,j);
                FN = sum(pat_conmat_inst(j,:)) - TP;
                FP = sum(pat_conmat_inst(:,j)) - TP;
                TN = sum(pat_conmat_inst(:)) - TP - FN - FP;
                pat_spec(1,j) = TN / (TN + FP);
            end
        end

    end

    if single_patient_testing
        pat_conmat = confusionmat(Yi, 1 * (pat_accy > 0.5), 'Order', classes-1);
        pat_sens(1,:) = diag(pat_conmat).' ./ (sum(pat_conmat, 2) + eps).';
        for j = 1:length(classes)
            TP = pat_conmat(j,j);
            FN = sum(pat_conmat(j,:)) - TP;
            FP = sum(pat_conmat(:,j)) - TP;
            TN = sum(pat_conmat(:)) - TP - FN - FP;
            pat_spec(1,j) = TN / (TN + FP);
        end
    end

    % Respecify variables
    metrics_add.win.conmat = win_conmat;
    metrics_add.win.spec = win_spec;
    metrics_add.win.sens = win_sens;
    metrics_add.win.accy = win_accy;
    metrics_add.pat.conmat = pat_conmat;
    metrics_add.pat.spec = pat_spec;
    metrics_add.pat.sens = pat_sens;
    metrics_add.pat.accy = pat_accy;

    % Write to database
    switch save_data.priority
        case "mysql"
            if save_data.save_mysql
                try
                    mysql_write(conn,table_name,parameters,1,metrics_add);
                catch
                    conn = mysql_login(conn.DataSource);
                    mysql_write(conn,table_name,parameters,1,metrics_add);
                end
            end
            if save_data.save_excel
                T = mysql_load(conn,table_name,"*");
                excel_path = save_data.mysql_excel_path;
                writetable(T, excel_path);
            end
        case "local"
            if save_data.save_excel
                excel_path = save_data.excel_path;
                local_write(excel_path,parameters,1,metrics_add);
            end
            if save_data.save_mysql
                try
                    mysql_write(conn,table_name,parameters,1,metrics_add);
                catch
                    conn = mysql_login(conn.DataSource);
                    mysql_write(conn,table_name,parameters,1,metrics_add);
                end
            end
    end

end
