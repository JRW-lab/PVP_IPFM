function gen_roc(result_parameters_hashes,line_configs,figure_data)

% Figure settings
load_path = "Models/";
figures_folder = 'Figures';
line_val = 2;
mark_val = 10;
font_val = 16;
marker_count = 6;
xlabel_name = "1 - Specificity";
ylabel_name = "Sensitivity";
loc = figure_data.legend_loc;
xlim_vec = figure_data.xlim_vec;
ylim_vec = figure_data.ylim_vec;
marker_space = linspace(0,max(xlim_vec),marker_count);

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
    try
        model = load((fullfile(load_path, sprintf("model_%s.mat",result_parameters_hashes(i)))));
    catch
        continue
    end
    test_probs = model.test_probs;
    true_labels = model.true_labels;

    % Temporary conversion - remove after all old data is gone
    if ~iscell(test_probs)
        test_probs = {test_probs};
    end

    % Create data vector of all iterations
    N = length(test_probs);
    probs_all = vertcat(test_probs{:});
    true_labels = repmat(true_labels,N,1);

    if size(probs_all,2) > 1
        % Amalgamate all data if doing multi-class detection
        true_labels_all = zeros(size(probs_all,1),size(probs_all,2));
        for m = 1:size(probs_all,2)
            true_labels_all(:,m) = true_labels == (m-1);
        end
    else
        true_labels_all = true_labels;
    end

    % Get ROC curve
    [X,Y] = perfcurve(true_labels_all(:), probs_all(:), 1);
    [~,marker_indices] = arrayfun(@(x) min(abs(X-x)), marker_space);
    plot(X,Y, ...
        line_styles{i}, ...
        Color=line_colors{i}, ...
        LineWidth=line_val, ...
        MarkerSize=mark_val, ...
        MarkerIndices=marker_indices)
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