function updateProgressBar(d)

% Import information
params = d;
model_settings = d.model_settings;
params = rmfield(params,"labels");
params = rmfield(params,"data_groups");
params = rmfield(params,"profile_sel");
params = rmfield(params,"configs");
params = rmfield(params,"primary_vals");
params = rmfield(params,"sel");
params = rmfield(params,"num_iters");
params = rmfield(params,"iter");
params = rmfield(params,"primvar_sel");
params = rmfield(params,"prvr_len");
params = rmfield(params,"conf_len");
params = rmfield(params,"current_frames");
params = rmfield(params,"num_frames");
params = rmfield(params,"model_settings");
settings_fields = fields(model_settings);
for i = 1:length(settings_fields)
    if isfield(params,settings_fields{i})
        params = rmfield(params,settings_fields{i});
    end
end
switch model_settings.model_type
    case "lrm"
        model_name = "logistic regression";
    case "tree"
        model_name = "random forest";
    case "nn-mlp"
        model_name = "multilayer perceptron neural network";
    case "nn-cnn1d"
        model_name = "1D-CNN";
end


% Create variables
config_count = (d.primvar_sel - 1) * d.conf_len + d.sel;
sim_count = config_count + (d.iter - 1) * d.prvr_len * d.conf_len;
config_length = d.prvr_len * d.conf_len;
sim_length = d.num_iters * config_length;

% Set up bar
pct = (sim_count / sim_length) * 100;
bar_len = 50;
filled_len = round(bar_len * sim_count / sim_length);
bar = [repmat('=', 1, filled_len), repmat(' ', 1, bar_len - filled_len)];

% Clear screen and print formatted simulation status
clc;
fprintf("RUNNING PROFILE %d - ITERATION %d of %d\n",d.profile_sel,d.iter,d.num_iters)
fprintf("(%d/%d) Training/testing %s model with config %d/%d for range value %d/%d\n", ...
    (d.primvar_sel-1)*length(d.configs) + d.sel, ...
    length(d.configs)*length(d.primary_vals), ...
    model_name,...
    d.sel,d.conf_len,...
    d.primvar_sel,d.prvr_len)

% Print current settings
data_fields = fields(params);
for i = 1:length(data_fields)
    fprintf("    %s = %s\n",data_fields{i},string(params.(data_fields{i})));
end

% Print grouping
fprintf("\n        Groups for model (comma separated):\n");
label_fields = fields(d.labels);
for i = 1:length(label_fields)
    fprintf("        %s: ", label_fields{i});
    for j = 1:length(d.data_groups)
        if ~ismissing(d.labels.(label_fields{i}))
            data_vec = string(d.labels.(label_fields{i}));
        else
            data_vec = string(d.data_groups{j}.(label_fields{i}));
        end
        if length(data_vec) > 1
            data_vec = strjoin(data_vec, '/');
        end

        fprintf("%s", data_vec);

        if j ~= length(d.data_groups)
            fprintf(", ")
        else
            fprintf("\n")
        end
    end
end

% Print progress bar
fprintf("\nProgress: [%s] %3.0f%% (%d/%d)\n\n", ...
    bar, pct, sim_count, sim_length);
end