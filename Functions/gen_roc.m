function gen_roc(result_parameters_hashes,line_configs,figure_data)

% Figure settings
load_path = "Models/";
figures_folder = 'Figures';
line_val = 2;
mark_val = 10;
font_val = 16;
marker_count = 7;
xlabel_name = "1 - Specificity";
ylabel_name = "Sensitivity";
loc = figure_data.legend_loc;
xlim_vec = figure_data.xlim_vec;
ylim_vec = figure_data.ylim_vec;
marker_space = linspace(0,max(xlim_vec),marker_count);
bootstrap_num = 1000;

% Import parameters
data_type = figure_data.data_type;
legend_vec = figure_data.legend_vec;
line_styles = figure_data.line_styles;
line_colors = figure_data.line_colors;
save_sel = figure_data.save_sel;

% Define multi-class styles
line_styles_multi = {"-o","-square","-diamond","-^","-v","-<","->"};

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
auc_bootsamps = zeros(bootstrap_num,length(line_configs));
AUC = zeros(length(line_configs),1);
CI = zeros(length(line_configs),2);
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

    if size(probs_all,2) > 2
        % Define number of labels
        num_labels = size(probs_all,2);

        figure(i)
        hold on

        legend_cell = cell(num_labels,1);
        hex = char(line_colors{i});
        rgb = sscanf(hex(2:end), '%2x%2x%2x', [1 3]) / 255;
        for j = 1:num_labels

            scale = 1 - (j-1) / num_labels;
            new_rgb = rgb * scale;

            true_labels_inst = true_labels == j-1;
            probs_inst = probs_all(:,j);
            [X,Y] = perfcurve(true_labels_inst, probs_inst, 1);
            [~,marker_indices] = arrayfun(@(x) min(abs(X-x)), marker_space);
            plot(X,Y, ...
                line_styles_multi{j}, ...
                Color=new_rgb, ...
                LineWidth=line_val, ...
                MarkerSize=mark_val, ...
                MarkerIndices=marker_indices)

            legend_cell{j} = sprintf("Y = %d",j-1);
        end

        % Set figure settings
        xlabel(xlabel_name)
        xlim(xlim_vec)
        ylabel(ylabel_name)
        ylim(ylim_vec)
        grid on
        legend(legend_cell,Location=loc);
        set(gca, 'FontSize', font_val);

        % Save figure
        timestamp = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
        timestamp_str = char(timestamp);
        figure_filename = fullfile(subsubfolder, "Figure_" + j + "_" + timestamp_str + ".png");
        if save_sel
            saveas(figure(i), figure_filename);
        end

    else
        % Get ROC curve
        figure(1)
        if i == 1
            hold on
        end
        [X,Y] = perfcurve(true_labels, probs_all(:,2), 1);
        [~,marker_indices] = arrayfun(@(x) min(abs(X-x)), marker_space);
        plot(X,Y, ...
            line_styles{i}, ...
            Color=line_colors{i}, ...
            LineWidth=line_val, ...
            MarkerSize=mark_val, ...
            MarkerIndices=marker_indices)

        % Gather bootstrap data for confidence interval of AUC
        n = length(true_labels);
        for j = 1:bootstrap_num
            idx = randsample(n,n,true);

            yb = true_labels(idx);
            pb = probs_all(idx,2);

            [~,~,~,auc_bootsamps(j,i)] = perfcurve(yb,pb,1);
        end

        % Compute AUC and CI
        AUC(i) = trapz(X,Y);
        CI(i,:) = prctile(auc_bootsamps(:,i),[2.5 97.5]);

        % Get calibration plot
        pts = 11;
        edges = linspace(0,1,pts);
        [~,~,bin] = histcounts(probs_all(:,2),edges);

        pred = zeros(pts-1,1);
        obs = zeros(pts-1,1);
        for j=1:pts-1
            idx = bin==j;
            pred(j) = mean(probs_all(idx,2));
            obs(j) = mean(true_labels(idx));
        end
        figure(2)
        if i == 1
            hold on
        end
        plot(pred(~isnan(pred)),obs(~isnan(obs)), ...
            line_styles{i}, ...
            Color=line_colors{i}, ...
            LineWidth=line_val, ...
            MarkerSize=mark_val)
        % plot([0 1],[0 1],'k--');

        if i == 3
            1;
        end


    end

end

% % Add a line for the random estimator
% plot([0 1],[0 1], "-.",LineWidth=line_val,Color="#4589ff")

% Show ROC figure
if size(probs_all,2) == 2

    % Set figure settings
    figure(1)
    % plot([0 1], [0 1], "--k",LineWidth=line_val)
    xlabel(xlabel_name)
    xlim(xlim_vec)
    ylabel(ylabel_name)
    ylim(ylim_vec)
    grid on
    legend(legend_vec,Location=loc);
    set(gca, 'FontSize', font_val);
    set(gca, 'Box', 'on');
    set(gca, 'LineWidth', line_val);

    % Save figure
    timestamp = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
    timestamp_str = char(timestamp);
    figure_filename = fullfile(subsubfolder, "Figure_" + timestamp_str + ".png");
    if save_sel
        saveas(figure(1), figure_filename);
    end

    % Set figure settings
    figure(2)
    xlabel('Predicted Probability');
    ylabel('Observed Frequency');
    xlim([0 1])
    ylim([0 1])
    grid on
    legend_ext = legend_vec;
    legend_ext{end+1} = "Ideal Est.";
    plot([0 1], [0 1], "--k",LineWidth=line_val)
    legend(legend_ext,Location=loc);
    set(gca, 'FontSize', font_val);
    set(gca, 'Box', 'on');
    set(gca, 'LineWidth', line_val);

    % Save figure
    timestamp = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
    timestamp_str = char(timestamp);
    figure_filename = fullfile(subsubfolder, "Figure_" + timestamp_str + ".png");
    if save_sel
        saveas(figure(2), figure_filename);
    end

end

% Create AUC/CI results table
data_table = struct2table([line_configs{:}]);
data_table.AUC = AUC;
data_table.CI_lower = CI(:,1);
data_table.CI_upper = CI(:,2);

% Display table
disp(data_table)

1;

%% SIGNIFICANCE TESTS ACROSS MODELS
% Compare two AUCs statistically.
% 
% Example:
% 
% Model A AUC = 0.84
% Model B AUC = 0.79
% p = 0.03
% 
% Then improvement is statistically significant.
% 
% If MATLAB lacks built-in DeLong
% 
% Use:
% 
% Option A: Bootstrap difference test
% 
% Compute:
% 
% AUC_A - AUC_B
% 
% across 1000 bootstrap samples.
% 
% If CI excludes zero → significant.

%% CALIBRATION PLOT

