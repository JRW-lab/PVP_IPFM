function sim_head(app_settings)
% This file loads data from the human and pig dataset to test the accuracy
% of a Elastic-net/ordinal logistical regression model, according to class
% division set within each profile.
%
% Coded 6/9/2025, JRW
% clc;

% Settings
dbname     = 'med_database';
% table_name = "bleeding_pigs_v1";
% table_name = "lrm_results_v2";
% delete_sel = true;
delete_sel = false;

% Import settings from matlab app
table_name = string(app_settings.table_name);
data_view = app_settings.data_view;
level_view = app_settings.level_view;
data_type = app_settings.data_type;
max_freq = app_settings.max_freq;
use_parellelization = app_settings.use_parellelization;
save_data.priority = app_settings.priority;
save_data.save_excel = app_settings.save_excel;
save_data.save_mysql = app_settings.save_mysql;
create_database_tables = app_settings.create_database_tables;
profile_sel = app_settings.profile_sel;
num_frames = app_settings.num_frames;



% Introduce, set up connection to MySQL server
addpath(fullfile(pwd, 'Common Functions'));
addpath(fullfile(pwd, 'Functions'));
javaaddpath('mysql-connector-j-8.4.0.jar');
save_data.excel_folder = 'Data';
save_data.excel_name = table_name;
save_data.excel_path = fullfile(save_data.excel_folder,save_data.excel_name + ".xlsx");
save_data.mysql_excel_path = fullfile(save_data.excel_folder,save_data.excel_name + "_mysqlbackup.xlsx");

% Set number of frames per iteration
render_figure = true;
save_sel = true;

% Preliminary Setup
if ~isfolder(save_data.excel_folder)
    mkdir(save_data.excel_folder);
end
if ~isfile(save_data.excel_path)
    % Create an empty Excel file
    writematrix([], save_data.excel_path);
end

% Extract data from profile
all_profiles = saved_profiles();
p_sel = all_profiles{profile_sel};
fields_names = fieldnames(p_sel);
for i = 1:numel(fields_names)
    eval([fields_names{i} ' = p_sel.(fields_names{i});']);
end
figure_data.ylim_vec = ylim_vec;
figure_data.legend_loc = legend_loc;
figure_data.xlim_vec = xlim_vec;
figure_data.ylim_vec = ylim_vec;
figure_data.data_type = data_type;
figure_data.legend_vec = legend_vec;
figure_data.line_styles = line_styles;
figure_data.line_colors = line_colors;
figure_data.save_sel = true;

% Set up ranges
if data_view == "figure"
    % data_type = "accy";
    if table_name == "lrm_results_v2"
        primary_var = "frequency_limit";
    else
        primary_var = p_sel.figure_var;
    end
    if primary_var == "frequency_limit"
        primary_vals = 5:5:max_freq;
    else
        primary_vals = p_sel.primary_vals;
    end
else
    primary_var = "frequency_limit";
    primary_vals = max_freq;
end

% Set up figure data
figure_data.level_view = level_view;
figure_data.data_type = data_type;
figure_data.primary_var = primary_var;
figure_data.primary_vals = primary_vals;
figure_data.legend_vec = legend_vec;
figure_data.line_colors = line_colors;
figure_data.save_sel = false;

%% Simulation setup

% Set up connection to MySQL server
if save_data.save_mysql
    conn = mysql_login(dbname);
    if create_database_tables
        % Set up MySQL commands
        sql_table = [
            "CREATE TABLE " + table_name + " (" ...
            "param_hash CHAR(64), " ...
            "parameters JSON, " ...
            "metrics JSON, " ...
            "frames_simulated INT NOT NULL, " ...
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, " ...
            "PRIMARY KEY (param_hash, frames_simulated)" ...
            ");"
            ];
        sql_flags = [
            "CREATE TABLE system_flags (" ...
            "id INT AUTO_INCREMENT PRIMARY KEY, " ...
            "flag_value TINYINT(1) DEFAULT 0" ...
            ");"
            ];
        sql_main_flag = "INSERT INTO system_flags (id, flag_value) VALUES (0, 0);";

        % Execute commands
        try
            execute(conn, join(sql_table));
        catch
        end
        try
            execute(conn, join(sql_flags));
        catch
        end
        try
            execute(conn, join(sql_main_flag));
        catch
        end
    end
else
    conn = [];
    if create_database_tables

    end
end

% Check already-saved results
switch save_data.priority
    case "mysql"
        if save_data.save_mysql
            T = mysql_load(conn,table_name,"*");
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
            T = mysql_load(conn,table_name,"*");
        end
end

% Find function files, get parameter list, modify sim data as needed
prvr_len = length(primary_vals);
conf_len = length(line_configs);

% Create result hashes
result_parameters_cell = cell(prvr_len,conf_len);
result_parameters_hashes = strings(prvr_len,conf_len);
prior_frames = zeros(length(primary_vals),length(line_configs));
mergestructs = @(x,y) cell2struct([struct2cell(x);struct2cell(y)],[fieldnames(x);fieldnames(y)]);
lc = line_configs;
for primvar_sel = 1:prvr_len

    % Set primary variable
    primvar_val = primary_vals(primvar_sel);

    % Go through each settings profile
    for sel = 1:conf_len

        % Set configuration
        config_sel = lc{sel};

        % Create overall parameters
        model_parameters_inst = model_parameters;
        model_parameters_inst.(primary_var) = primvar_val;
        result_parameters = mergestructs(data_defaults,model_parameters_inst);

        % Overwrite settings with config setting
        config_fields = fields(config_sel);
        for i = 1:length(config_fields)
            if isfield(result_parameters,config_fields{i})
                result_parameters.(config_fields{i}) = config_sel.(config_fields{i});
            elseif isfield(result_parameters.labels,config_fields{i})
                result_parameters.labels.(config_fields{i}) = config_sel.(config_fields{i});
            else
                error("Line configuration contains invalid fields")
            end
        end

        % Remove defaults that are being overwritten
        used_labels = unique(string(cellfun(@(x) fields(x), data_groups, "UniformOutput", false)));
        for i = 1:length(used_labels)
            result_parameters.labels.(used_labels(i)) = NaN;
        end

        % Generate result hash
        result_parameters.data_groups = data_groups;
        [~,paramHash] = jsonencode_sorted(result_parameters);

        % Save to stack
        result_parameters_cell{primvar_sel,sel} = result_parameters;
        result_parameters_hashes(primvar_sel,sel) = paramHash;

        % Either delete the saved data and reset, or note previous progress
        if delete_sel && ismember(sel,delete_configs)
            % Delete data from database/table
            switch save_data.priority
                case "mysql"
                    if save_data.save_mysql
                        delete_command = sprintf("DELETE FROM %s WHERE param_hash = '%s';",table_name,paramHash);
                        exec(conn, delete_command);
                    elseif save_data.save_excel
                        table_locs = 1 - (string(T.param_hash) == paramHash);
                        T = T(logical(table_locs),:);
                    end
                case "local"
                    if save_data.save_excel
                        table_locs = 1 - (string(T.param_hash) == paramHash);
                        T = T(logical(table_locs),:);
                    elseif save_data.save_mysql
                        delete_command = sprintf("DELETE FROM %s WHERE param_hash = '%s';",table_name,paramHash);
                        exec(conn, delete_command);
                    end
            end
        else
            % Load data from DB
            try
                sim_result = T(string(T.param_hash) == paramHash, :);
                prior_frames(primvar_sel,sel) = sim_result.frames_simulated;
            catch
                prior_frames(primvar_sel,sel) = 0;
            end
        end

    end
end

if sum(prior_frames >= num_frames,"all") == length(primary_vals) * length(line_configs)
    skip_simulations = true;
else
    skip_simulations = false;
end

%% Simulation loop

if ~skip_simulations
    % Set up connection to MySQL server
    if use_parellelization
        if isempty(gcp('nocreate'))
            poolCluster = parcluster('local');
            maxCores = poolCluster.NumWorkers;  % Get the max number of workers available
            parpool(poolCluster, maxCores);     % Start a parallel pool with all available workers
        end
        parfevalOnAll(@() javaaddpath('mysql-connector-j-8.4.0.jar'), 0);
        projectPath = fullfile(pwd);
        addpath(genpath(projectPath));
        parfevalOnAll(@() addpath(genpath(projectPath)), 0);
    end

    % Progress tracking setup
    dq = parallel.pool.DataQueue;
    afterEach(dq, @updateProgressBar);

    for iter = 1:num_frames

        if use_parellelization

            % Go through each settings profile
            parfor primvar_sel = 1:prvr_len


                for sel = 1:conf_len

                    % Continue to simulate if need more frames
                    if iter > prior_frames(primvar_sel,sel)

                        % Set connection
                        conn_thrall = mysql_login(dbname);

                        % Set delete condition
                        if delete_sel && ismember(sel,delete_configs)
                            delete_model = true;
                        else
                            delete_model = false;
                        end

                        % Select parameters
                        result_parameters_inst = result_parameters_cell{primvar_sel,sel};
                        result_hash_inst = result_parameters_hashes(primvar_sel,sel);
 
                        % Print message
                        progress_bar_data = result_parameters_inst;
                        progress_bar_data.profile_sel = profile_sel;
                        progress_bar_data.configs = lc;
                        progress_bar_data.primary_vals = primary_vals;
                        progress_bar_data.sel = sel;
                        progress_bar_data.num_iters = num_frames;
                        progress_bar_data.iter = iter;
                        progress_bar_data.primvar_sel = primvar_sel;
                        progress_bar_data.sel = sel;
                        progress_bar_data.prvr_len = prvr_len;
                        progress_bar_data.conf_len = conf_len;
                        progress_bar_data.current_frames = iter;
                        progress_bar_data.num_frames = num_frames;
                        send(dq, progress_bar_data);

                        % Simulate under current settings
                        model_fun_v3(save_data,conn_thrall,table_name,result_parameters_inst,result_hash_inst,iter,delete_model)

                        % Close connection instance
                        close(conn_thrall)

                    end
                end

            end
        else

            % Go through each settings profile
            for primvar_sel = 1:prvr_len
                for sel = 1:conf_len

                    % Select parameters
                    result_parameters_inst = result_parameters_cell{primvar_sel,sel};
                    result_hash_inst = result_parameters_hashes(primvar_sel,sel);

                    % Continue to simulate if need more frames
                    if iter > prior_frames(primvar_sel,sel)

                        % Set delete condition
                        if delete_sel && ismember(sel,delete_configs)
                            delete_model = true;
                        else
                            delete_model = false;
                        end

                        % Print message
                        progress_bar_data = result_parameters_inst;
                        progress_bar_data.profile_sel = profile_sel;
                        progress_bar_data.configs = line_configs;
                        progress_bar_data.primary_vals = primary_vals;
                        progress_bar_data.sel = sel;
                        progress_bar_data.num_iters = num_frames;
                        progress_bar_data.iter = iter;
                        progress_bar_data.primvar_sel = primvar_sel;
                        progress_bar_data.sel = sel;
                        progress_bar_data.prvr_len = prvr_len;
                        progress_bar_data.conf_len = conf_len;
                        progress_bar_data.current_frames = iter;
                        progress_bar_data.num_frames = num_frames;
                        send(dq, progress_bar_data);

                        % Simulate under current settings
                        model_fun_v3(save_data,conn,table_name,result_parameters_inst,result_hash_inst,iter,delete_model)

                    end
                end
            end
        end
    end
end

%% Figure generation

% Generate figure
clc;
fprintf("Displaying results for profile %d:\n",profile_sel)
if render_figure
    figure_data.save_sel = save_sel;
    switch data_view
        case "table"
            % Generate table
            gen_table(save_data,conn,table_name,result_parameters_hashes,line_configs,figure_data);
        case "figure"
            clf
            % Generate figure
            gen_figure_v2(save_data,conn,table_name,result_parameters_hashes,line_configs,figure_data);
        case "roc"
            clf
            % Generate ROC curve
            gen_roc(result_parameters_hashes,line_configs,figure_data);
    end
end