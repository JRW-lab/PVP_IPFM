function gen_table(save_data,conn,table_name,result_parameters_hashes,line_configs,figure_data)

% Import Parameters
data_type = figure_data.data_type;
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
conmat_tables = cell(length(line_configs),1);
for sel = 1:length(line_configs)

    % Load data from parameter hash
    paramHash = result_parameters_hashes(1,sel);
    sim_result = T(string(T.param_hash) == paramHash, :);

    % Select data to extract
    results_inst = jsondecode(sim_result.metrics{1});
    results_vec{sel} = results_inst.(level_view).(data_type);
    conmat_tables{sel} = results_inst.(level_view).conmat;

    % Normalize confusion matrices
    conmat_tables{sel} = conmat_tables{sel} ./ sum(conmat_tables{sel},2);

end

% Get results name for column
results_name = sprintf("%s.%s",level_view,data_type);

% Create results table
data_table = struct2table([line_configs{:}]);
data_table.(results_name) = results_vec;

% Display table
disp(data_table)

% Display confusion matrices
for i = 1:numel(conmat_tables)
    fprintf('Confusion Matrix %d:\n', i);
    disp(conmat_tables{i});
end