function gen_table(save_data,conn,table_name,result_parameters_hashes,line_configs,figure_data)


% Settings
line_val = 2;
mark_val = 10;
font_val = 16;

% Import Parameters
figures_folder = 'Figures';
loc = figure_data.legend_loc;
ylim_vec = figure_data.ylim_vec;
data_type = figure_data.data_type;
primary_var = figure_data.primary_var;
primary_vals = figure_data.primary_vals;
legend_vec = figure_data.legend_vec;
line_styles = figure_data.line_styles;
line_colors = figure_data.line_colors;
save_sel = figure_data.save_sel;
level_view = figure_data.level_view;

% Load data from DB and set new frame count
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

% Go through each settings profile
results_vec = cell(length(line_configs),1);
for sel = 1:length(line_configs)

    % Load data from parameter hash
    paramHash = result_parameters_hashes(1,sel);
    sim_result = T(string(T.param_hash) == paramHash, :);

    % Select data to extract
    results_inst = jsondecode(sim_result.metrics{1});
    results_val = results_inst.(level_view).(data_type);
    % results_val = mean(results_val);

    % Add data to stack
    results_vec{sel} = results_val.';

end

% figure(1)
% subplot(3,1,1)
% histogram(results_vec{1},10,"FaceColor","red")
% ylim([0 11])
% ylabel("Frequency")
% % yticks(0:2:10)
% grid on;
% set(gca, 'FontSize', font_val);
% legend("Raw PVP",Location="northwest")
% subplot(3,1,2)
% histogram(results_vec{3},10)
% ylim([0 11])
% ylabel("Frequency")
% % yticks(0:2:10)
% grid on;
% set(gca, 'FontSize', font_val);
% legend("IPFM-PVP",Location="northwest")
% subplot(3,1,3)
% histogram(results_vec{5},10,"FaceColor","blue")
% ylim([0 11])
% ylabel("Frequency")
% % yticks(0:2:10)
% grid on;
% set(gca, 'FontSize', font_val);
% legend("IPFM-EHR",Location="northwest")
% xlabel("LOOECV Model Accuracy")
% 
% figure(2)
% plot([results_vec{1}; results_vec{3}; results_vec{5}])
% xticks(1:1:3)
% xticklabels(["Raw PVP" "IPFM-PVP" "IPFM-EHR"])
% ylabel("LOOVE Model Accuracy")
% grid on;
% set(gca, 'FontSize', font_val);

% Get results name for column
results_name = sprintf("%s.%s",level_view,data_type);

% Create results table
data_table = struct2table([line_configs{:}]);
data_table.(results_name) = results_vec;

% Display table
disp(data_table)