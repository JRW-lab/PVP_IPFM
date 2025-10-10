function gen_roc(result_parameters_hashes,line_configs,figure_data)

% Figure settings
load_path = "Models/";
figures_folder = 'Figures';
line_val = 2;
mark_val = 10;
font_val = 16;
xlabel_name = "1 - Specificity";
ylabel_name = "Sensitivity";
loc = figure_data.legend_loc;
xlim_vec = figure_data.xlim_vec;
ylim_vec = figure_data.ylim_vec;

% Import parameters
data_type = figure_data.data_type;
legend_vec = figure_data.legend_vec;
line_styles = figure_data.line_styles;
line_colors = figure_data.line_colors;
save_sel = figure_data.save_sel;

% Create folders if they don't exist
subfolder = fullfile(figures_folder, ['/' char(data_type)]);
subsubfolder = fullfile(subfolder, '/ROC');
if ~exist(figures_folder, 'dir')
    mkdir(figures_folder);
end
if ~exist(subfolder, 'dir')
    mkdir(subfolder);
end
if ~exist(subsubfolder, 'dir')
    mkdir(subsubfolder);
end

% Plot figure
figure(1)
hold on
for i = 1:length(line_configs)

    % Load model and data
    model = load((fullfile(load_path, sprintf("model_%s.mat",result_parameters_hashes(i)))));
    test_probs = model.test_probs;
    true_labels = model.true_labels;

    % Create data vector of all iterations
    N = size(test_probs,2);
    probs_all = test_probs(:); 
    true_labels_all = repmat(true_labels, N, 1);
    
    % Get ROC curve
    [X,Y] = perfcurve(true_labels_all, probs_all, 1);
    plot(X,Y, ...
        line_styles{i}, ...
        Color=line_colors{i}, ...
        LineWidth=line_val, ...
        MarkerSize=mark_val)
end

% Set figure settings
xlabel(xlabel_name)
xlim(xlim_vec)
ylabel(ylabel_name)
ylim(ylim_vec)
grid on
legend(legend_vec,Location=loc);
set(gca, 'FontSize', font_val);

% Save figure
timestamp = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
timestamp_str = char(timestamp);
figure_filename = fullfile(subsubfolder, "Figure_" + timestamp_str + ".png");
if save_sel
    saveas(figure(1), figure_filename);
end