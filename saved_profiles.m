function [all_profiles,profile_names] = saved_profiles()

% Initialize cell array of profiles
all_profiles = cell(0);
profile_names = cell(0);

%% PROFILE 1
profile_name = "Human, R/H - 5/10s Windows (All Patients)";
p = struct;
p.data_defaults = struct(...
    'dataset', "Human", ...
    'signal_sel', "raw_signal", ...
    'labels', struct("bolus_type", "BB", ...
    "hypovolemic", "R"), ...
    'exclude_patients', []);
p.data_groups = {
    struct("hypovolemic", "R")
    struct("hypovolemic", "H")
    };
p.model_parameters = struct(...
    'window_duration', 10, ...
    'frequency_limit', 30, ...
    'alpha', 0.5, ...
    'training_type', "percentage", ...
    'split_by', "signal", ...
    'tshift', 1, ...
    'drop_rho_below', 0, ...
    'drop_DC_component', true, ...
    'normalize_power', true);
p.line_configs = {
    struct('signal_sel', "raw_signal", 'window_duration', 10)
    struct('signal_sel', "raw_signal", 'window_duration', 5)
    struct('signal_sel', "IPFM_signal", 'window_duration', 10)
    struct('signal_sel', "IPFM_signal", 'window_duration', 5)
    struct('signal_sel', "EHR_signal", 'window_duration', 10)
    struct('signal_sel', "EHR_signal", 'window_duration', 5)
    };
p.legend_vec = {
    "Raw PVP, 10s windows"
    "Raw PVP, 5s windows"
    "IPFM-PVP, 10s windows"
    "IPFM-PVP, 5s windows"
    "IPFM-EHR, 10s windows"
    "IPFM-EHR, 5s windows"
    };
p.line_styles = {
    "-o"
    "--x"
    "-o"
    "--x"
    "-o"
    "--x"
    };
p.line_colors = {
    "#FF0000"
    "#FF0000"
    "#0F62FE"
    "#0F62FE"
    "#24A249"
    "#24A249"
    };
p.delete_configs = [];
p.legend_loc = "southeast";
p.xlim_vec = [0 0.3];
p.ylim_vec = [0.6 1];
all_profiles = [all_profiles p];
profile_names = [profile_names profile_name];

%% PROFILE 2
profile_name = "Human, R/H - 5/10s Windows (rho >= 0.9)";
p = struct;
p.data_defaults = struct(...
    'dataset', "Human", ...
    'signal_sel', "raw_signal", ...
    'labels', struct("bolus_type", "BB", ...
    "hypovolemic", "R"), ...
    'exclude_patients', []);
p.data_groups = {
    struct("hypovolemic", "R")
    struct("hypovolemic", "H")
    };
p.model_parameters = struct(...
    'window_duration', 10, ...
    'frequency_limit', 30, ...
    'alpha', 0.5, ...
    'training_type', "percentage", ...
    'split_by', "signal", ...
    'tshift', 1, ...
    'drop_rho_below', 0.9, ...
    'drop_DC_component', true, ...
    'normalize_power', true);
p.line_configs = {
    struct('signal_sel', "raw_signal", 'window_duration', 10)
    struct('signal_sel', "raw_signal", 'window_duration', 5)
    struct('signal_sel', "IPFM_signal", 'window_duration', 10)
    struct('signal_sel', "IPFM_signal", 'window_duration', 5)
    struct('signal_sel', "EHR_signal", 'window_duration', 10)
    struct('signal_sel', "EHR_signal", 'window_duration', 5)
    };
p.legend_vec = {
    "Raw PVP, 10s windows"
    "Raw PVP, 5s windows"
    "IPFM-PVP, 10s windows"
    "IPFM-PVP, 5s windows"
    "IPFM-EHR, 10s windows"
    "IPFM-EHR, 5s windows"
    };
p.line_styles = {
    "-o"
    "--x"
    "-o"
    "--x"
    "-o"
    "--x"
    };
p.line_colors = {
    "#FF0000"
    "#FF0000"
    "#0F62FE"
    "#0F62FE"
    "#24A249"
    "#24A249"
    };
p.delete_configs = [];
p.legend_loc = "southeast";
p.xlim_vec = [0 0.3];
p.ylim_vec = [0.6 1];
all_profiles = [all_profiles p];
profile_names = [profile_names profile_name];

%% PROFILE 3
profile_name = "Human, R/H - Patient split (rho >= 0.9)";
p = struct;
p.data_defaults = struct(...
    'dataset', "Human", ...
    'signal_sel', "raw_signal", ...
    'labels', struct("bolus_type", "BB", ...
    "hypovolemic", "R"), ...
    'exclude_patients', []);
p.data_groups = {
    struct("hypovolemic", "R")
    struct("hypovolemic", "H")
    };
p.model_parameters = struct(...
    'window_duration', 10, ...
    'frequency_limit', 30, ...
    'alpha', 0.5, ...
    'training_type', "percentage", ...
    'split_by', "patient", ...
    'randomize_training', true, ...
    'tshift', 1, ...
    'drop_rho_below', 0.9, ...
    'drop_DC_component', true, ...
    'normalize_power', true);
p.line_configs = {
    struct('signal_sel', "raw_signal")
    struct('signal_sel', "IPFM_signal")
    struct('signal_sel', "EHR_signal")
    };
p.legend_vec = {
    "Raw PVP"
    "IPFM-PVP"
    "IPFM-EHR"
    };
p.line_styles = {
    "-o"
    % "--x"
    "-o"
    % "--x"
    "-o"
    % "--x"
    };
p.line_colors = {
    "#FF0000"
    % "#FF0000"
    "#0F62FE"
    % "#0F62FE"
    "#24A249"
    % "#24A249"
    };
p.delete_configs = [1,2,3];
p.legend_loc = "northeast";
p.xlim_vec = [0 0.3];
p.ylim_vec = [0.5 1];
all_profiles = [all_profiles p];
profile_names = [profile_names profile_name];

%% PROFILE 4
profile_name = "Human, R/H - AB vs BB";
p = struct;
p.data_defaults = struct(...
    'dataset', "Human", ...
    'signal_sel', "raw_signal", ...
    'labels', struct("bolus_type", "BB", ...
    "hypovolemic", "R"), ...
    'exclude_patients', []);
p.data_groups = {
    struct("hypovolemic", "R")
    struct("hypovolemic", "H")
    };
p.model_parameters = struct(...
    'window_duration', 10, ...
    'frequency_limit', 30, ...
    'alpha', 0.5, ...
    'training_type', "percentage", ...
    'split_by', "signal", ...
    'tshift', 1, ...
    'drop_rho_below', 0, ...
    'drop_DC_component', true, ...
    'normalize_power', true);
p.line_configs = {
    struct('signal_sel', "raw_signal", 'bolus_type', "BB")
    struct('signal_sel', "raw_signal", 'bolus_type', "AB")
    struct('signal_sel', "IPFM_signal", 'bolus_type', "BB")
    struct('signal_sel', "IPFM_signal", 'bolus_type', "AB")
    struct('signal_sel', "EHR_signal", 'bolus_type', "BB")
    struct('signal_sel', "EHR_signal", 'bolus_type', "AB")
    };
p.legend_vec = {
    "Raw PVP, BB"
    "Raw PVP, AB"
    "IPFM-PVP, BB"
    "IPFM-PVP, AB"
    "IPFM-EHR, BB"
    "IPFM-EHR, AB"
    };
p.line_styles = {
    "-o"
    "--x"
    "-o"
    "--x"
    "-o"
    "--x"
    };
p.line_colors = {
    "#FF0000"
    "#FF0000"
    "#0F62FE"
    "#0F62FE"
    "#24A249"
    "#24A249"
    };
p.delete_configs = [];
p.legend_loc = "southeast";
p.xlim_vec = [0 0.3];
p.ylim_vec = [0.7 1];
all_profiles = [all_profiles p];
profile_names = [profile_names profile_name];
