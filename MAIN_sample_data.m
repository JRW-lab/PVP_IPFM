%% MAIN_sample_data.m
% This file loads data from the human and pig dataset to analyze the
% statical properties, according to class division set within each profile.
% 
% Coded 6/9/2025, JRW
%% Load data and startup
clear; clc; close all;

% Describe needed paths
addpath(fullfile(pwd, '\Functions'));
data_path = "Data";
figures_path = "Figures";

% Controls
profile_sel = 4;
line_val = 2;
mark_val = 10;
font_val = 16;

% % Specify valid patients
% BB_patients = [8,9,10,12,19,20,27,28,30,31,4,5,6,13,25,26,32,34];
% AB_patients = [19,20,27,28,30,31,4,5,6,13,25,26,34];

% Data select
switch profile_sel
    case 1 % PROFILE 1 - Human data, resuscitated vs hypovolemic, BB Data
        dataset = "Human";
        signal_type = "PVP";

        % Select group of data to examine
        group_type = "bolus_type";
        type_sel = "BB";

        % Select category to divide by
        group_category = "hypovolemic";
        null_group = "R";
        hypo_group = "H";
        group_value = "NA";
        exclude_patients = "NA";
        drop_rho_below = 0.0;

    case 2 % PROFILE 2 - Human data, resuscitated vs hypovolemic, AB Data
        dataset = "Human";
        signal_type = "PVP";

        % Select group of data to examine
        group_type = "bolus_type";
        type_sel = "AB";

        % Select category to divide by
        group_category = "hypovolemic";
        null_group = "R";
        hypo_group = "H";
        group_value = "NA";
        exclude_patients = ["P21"    "P22"    "P24"    "P32"    "P33"    "P35"    "P36"     "P2"     "P7"     "P9"    "P18"];
        drop_rho_below = 0;

    case 3 % PROFILE 3 - Pig data, MAC vs PRO
        dataset = "Pig";
        signal_type = "PVP";

        % Select group of data to examine
        group_type = "bleeding";
        type_sel = "S";

        % Select category to divide by
        group_category = "anesthetic_type";
        null_group = "PRO";
        hypo_group = "MAC";
        group_value = "anesthetic_level";
        null_val = 1:3;
        hypo_val = 1:3;
        exclude_patients = "NA";
        drop_rho_below = 0.9;

    case 4 % PROFILE 4 - Pig data, Stable vs Bleeding, PRO4
        dataset = "Pig";
        signal_type = "PVP";

        % Select group of data to examine
        group_type = "anesthetic_type";
        type_sel = "PRO";

        % Select category to divide by
        group_category = "bleeding";
        null_group = "S";
        hypo_group = "B";
        group_value = "anesthetic_level";
        null_val = 4;
        hypo_val = 4;
        exclude_patients = "NA";
        drop_rho_below = 0.9;

end

% Load lookup table and set parameters
load_path = data_path + "/" + dataset + "/";
load(fullfile(load_path, 'lookup_table.mat'), 'lookup_table');
fs = 1000;
t_shift = 1;
window_duration = 10;
frequency_limit = 30;
signal_sel = ["raw_signal","IPFM_signal","EHR_signal"];
signal_names = ["Raw","Synth.","EHR"];

% Loop through each entry in the lookup table
T_null = [];
T_hypo = [];
rho_null = [];
rho_hypo = [];
data_null = {};
data_hypo = {};
count_null = 0;
count_hypo = 0;
labels_null = [];
labels_hypo = [];
for i = 1:height(lookup_table)

    fprintf("Loading data file %d of %d...\n",i,height(lookup_table))

    type_inst = lookup_table.(group_type)(i);

    % Check group_type match
    if ~isequal(type_inst{1}, type_sel)
        continue; % Skip this entry
    end

    % Load the file
    filename = lookup_table.filename{i};
    file_path = fullfile(load_path, filename);
    S = load(file_path);  

    % Check for missing or invalid patient
    if ismember(S.data.name,exclude_patients) || ~isequal(S.labels.signal_type, signal_type) || S.data.rho <= drop_rho_below
        continue;
    end

    % Determine group membership based on group_category
    label_name = S.labels.(group_category);
    if group_value == "NA"
        label_value = NaN;
    else
        label_value = S.labels.(group_value);
    end

    % Set data in its place
    if isequal(label_name, null_group) && (group_value == "NA" || ismember(label_value, null_val))
        count_null = count_null + 1;
        T_null(count_null,1) = S.data.T;
        rho_null(count_null,1) = S.data.rho;
        for j = 1:length(signal_sel)
            data_null{count_null,j} = S.data.(signal_sel(j));
        end
        labels_null = [labels_null; S.labels];
    elseif isequal(label_name, hypo_group) && (group_value == "NA" || ismember(label_value, hypo_val))
        count_hypo = count_hypo + 1;
        T_hypo(count_hypo,1) = S.data.T;
        rho_hypo(count_hypo,1) = S.data.rho;
        for j = 1:length(signal_sel)
            data_hypo{count_hypo,j} = S.data.(signal_sel(j));
        end
        labels_hypo = [labels_hypo; S.labels];
    end
end

% Generate time-windows from data
fprintf("Generating time-windows from data...\n")
twindows_null = cellfun(@(x) make_twindows(x,fs,window_duration,t_shift*fs),data_null,"UniformOutput",false);
twindows_hypo = cellfun(@(x) make_twindows(x,fs,window_duration,t_shift*fs),data_hypo,"UniformOutput",false);

% Generate freq-windows from data
fprintf("Generating frequency-windows from data...\n")
fwindows_null = cellfun(@(x) fft_rhys(x,fs,frequency_limit,window_duration),twindows_null,'UniformOutput',false);
fwindows_hypo = cellfun(@(x) fft_rhys(x,fs,frequency_limit,window_duration),twindows_hypo,'UniformOutput',false);

%% Power Spectral Density

% Compute Power Spectral Density (PSD)
f_range = 0:1/window_duration:frequency_limit-1/window_duration;

% Set figure info
xlim_vec_psd = [0 5];
ylim_vec_psd = [-30 0];

% Set up figure
figure(1)
hold on

for k = 1:2

    % Select null/alternative hypothesis
    switch k
        case 1
            linecolor = "#da1e28";
            windows = fwindows_null(:,1);
        case 2
            linecolor = "#0f62fe";
            windows = fwindows_hypo(:,1);
    end

    % Create new frequency-separated samples
    block = vertcat(windows{:}).^2;
    psd = (sum(block,1) / size(block,1));
    plot(f_range,10*log10(psd),Color=linecolor,linewidth=line_val,LineStyle="-")

end

% Finish figure setup
grid on
xlim(xlim_vec_psd)
ylim(ylim_vec_psd)
xlabel("Frequency (Hz)")
ylabel("Power/Frequency (dB/Hz)")
if profile_sel == 1 || profile_sel == 2
    legend("Resuscitated","Hypovolemic")
elseif profile_sel == 3
    legend("PRO","MAC")
elseif profile_sel == 4
    legend("Stable","Bleeding")
end
set(gca, 'FontSize', font_val);

%% Empirical CDF

% CDF settings
if profile_sel == 1 || profile_sel == 2
    frequency_sel =  2.3;
    xlim_vec = [0 1];
    ylim_vec = [0 1];
elseif profile_sel == 3
    frequency_sel =  1.2;
    xlim_vec = [0 3];
    ylim_vec = [0 1];
elseif profile_sel == 4
    frequency_sel =  1.4;
    xlim_vec = [0 3.5];
    ylim_vec = [0 1];
end
index = round(frequency_sel * window_duration) + 1;

% Set up figure
figure(3)
hold on

for j = 1:length(signal_sel)
    for k = 1:2

        % Select color for signal type
        switch j
            case 1
                linecolor = "#da1e28";
            case 2
                linecolor = "#0f62fe";
            case 3
                linecolor = "#198038";
        end

        % Select null/alternative hypothesis
        switch k
            case 1
                linestyle = "-";
                windows = fwindows_null(:,j);
            case 2
                linestyle = "--";
                windows = fwindows_hypo(:,j);
        end

        % Create new frequency-separated samples
        block = vertcat(windows{:});

        % Get empirical CDF's
        datavec_sel = block(:,index);
        [F,x] = ecdf(datavec_sel);
        plot(x,F,Color=linecolor,linewidth=line_val,LineStyle=linestyle)

    end
end

% Finish figure setup
grid on;
xlim(xlim_vec)
ylim(ylim_vec)
xlabel("Signal amplitude (mmHg)")
ylabel("Probability")
set(gca, 'FontSize', font_val);
if profile_sel == 1 || profile_sel == 2
    legend("Raw-PVP, Resuscitated",...
        "Raw-PVP, Hypovolemic",...
        "IPFM-PVP, Resuscitated",...
        "IPFM-PVP, Hypovolemic",...
        "IPFM-HB, Resuscitated",...
        "IPFM-HB, Hypovolemic",...
        Location="southeast")
elseif profile_sel == 3
    legend("Raw-PVP, PRO",...
        "Raw-PVP, MAC",...
        "IPFM-PVP, PRO",...
        "IPFM-PVP, MAC",...
        "IPFM-HB, PRO",...
        "IPFM-HB, MAC",...
        Location="southeast")
elseif profile_sel == 4
    legend("Raw-PVP, Stable",...
        "Raw-PVP, Bleeding",...
        "IPFM-PVP, Stable",...
        "IPFM-PVP, Bleeding",...
        "IPFM-HB, Stable",...
        "IPFM-HB, Bleeding",...
        Location="southeast")
end

%% Canonical Correlation Analysis

p_val = zeros(3,1);
tbl = zeros(3,1);
r_tbl = zeros(3,1);
stats = cell(3,1);
for j = 1:length(signal_sel)

    % Separate resusitated and hypovolemic data
    resu_windows = fwindows_null(:,j);
    hypo_windows = fwindows_hypo(:,j);

    % Create new frequency-separated samples
    resu_block = vertcat(resu_windows{:});
    hypo_block = vertcat(hypo_windows{:});

    % Get sizes of data
    L = size(resu_block, 2); % Length of each window
    N = size(resu_block, 1); % Number of windows for Group 1
    M = size(hypo_block, 1); % Number of windows for Group 2

    % Combine data
    all_block = [resu_block; hypo_block];

    % Create labels for each window
    group_labels = [ones(N, 1); 2 * ones(M, 1)]; % Group 1 = 1, Group 2 = 2

    % Conduct MANOVA
    [p_val(j), tbl(j), stats{j}] = manova1(all_block, group_labels);

    % Conduct Canonical Correlation Analysis
    [A,B,r,U,V] = canoncorr(resu_block,hypo_block(1:2123,:));
    r_tbl(j) = r(1);

    t = tiledlayout(2,2);
    title(t,'Canonical Scores of X vs Canonical Scores of Y')
    xlabel(t,'Canonical Variables of X')
    ylabel(t,'Canonical Variables of Y')
    t.TileSpacing = 'compact';

    nexttile
    plot(U(:,1),V(:,1),'.')
    xlabel('u1')
    ylabel('v1')

    nexttile
    plot(U(:,2),V(:,1),'.')
    xlabel('u2')
    ylabel('v1')

    nexttile
    plot(U(:,1),V(:,2),'.')
    xlabel('u1')
    ylabel('v2')

    nexttile
    plot(U(:,2),V(:,2),'.')
    xlabel('u2')
    ylabel('v2')

end

fprintf("\nSample canonical correlations:\n")
disp(r_tbl)

%% KS Two-sample Test on Signal

% KS Settings
f_range = 0:1/window_duration:frequency_limit-1/window_duration;

% Set up figure
figure(4)
hold on

p_vals = cell(length(signal_sel),1);
ks_test_stat = cell(length(signal_sel),1);
log_p_vals = cell(length(signal_sel),1);
log_ks_test = cell(length(signal_sel),1);
mv_ks_tests = cell(1,length(signal_sel));
for j = 1:length(signal_sel)

    % Select data for each window
    switch j
        case 1
            linecolor = "#da1e28";
            linestyle = "-";
            linemarker = "o";
        case 2
            linecolor = "#0f62fe";
            linestyle = "-";
            linemarker = "^";
        case 3
            linecolor = "#198038";
            linestyle = "-";
            linemarker = "*";
    end

    % Create new frequency-separated samples
    resu_block = cell2mat(fwindows_null(:,j));
    hypo_block = cell2mat(fwindows_hypo(:,j));
    resu_samps = mat2cell(resu_block, size(resu_block,1), ones(1,size(resu_block,2)));
    hypo_samps = mat2cell(hypo_block, size(hypo_block,1), ones(1,size(hypo_block,2)));

    % Generate two-sample multivariate KS test statistic
    mv_ks_tests{j} = mv_kstest2(resu_block,hypo_block);

    % Test the samples using KS two-sample test
    [~,p_vals{j},ks_test_stat{j}] = cellfun(@(x,y) kstest2(x,y), resu_samps, hypo_samps);
    log_p_vals{j} = log10(p_vals{j});
    log_ks_test{j} = (ks_test_stat{j});
    log_p_vals{j} = max(log_p_vals{j},-400);

    % Plot result of KS test
    plot(f_range, log_ks_test{j}, Color=linecolor,LineStyle=linestyle,LineWidth=line_val);

end

% Finish Univariate KS figure setup
grid on
xlabel("Frequency (Hz)")
ylabel("Dist. between Empirical CDFs")
set(gca, 'FontSize', font_val);
legend("Raw-PVP","IPFM-PVP","IPFM-HB",Location="northeast")

clc
% Display results for multivariate KS test
figure(20)
hold on
grid on
mv_ks_vals = zeros(length(signal_sel),1);
for i = 1:length(signal_sel)

    % Select data for each window
    switch i
        case 1
            linecolor = "#da1e28";
            linestyle = "-";
            linemarker = "o";
        case 2
            linecolor = "#0f62fe";
            linestyle = "-";
            linemarker = "^";
        case 3
            linecolor = "#198038";
            linestyle = "-";
            linemarker = "*";
    end

    plot(mv_ks_tests{i}, Color=linecolor,LineStyle=linestyle,LineWidth=line_val)

    mv_ks_vals(i) = max(mv_ks_tests{i});
    fprintf("Two-Sample Multivariate KS Test Statistic for %s: %.4f\n",signal_sel(i),mv_ks_vals(i))
end
% plot([size(resu_block,1) size(resu_block,1)], [10000 10000])
set(gca, 'YScale', 'log')
xlabel("Sample from Z")
ylabel("Dist. between Empirical CDFs")
legend("Raw-PVP","IPFM-PVP","IPFM-HB",Location="northwest")




function D = mv_kstest2(X,Y)

% Import sizes and test compatibility
[nx,k] = size(X);
[ny,k2] = size(Y);
if k ~= k2
    error("x1 and x2 must have same number of features (columns).")
end

% Combine matrices
Z = [X; Y];
N = nx + ny;

% Find cumulative statistic for each feature
D = zeros(k,1);
for i = 1:N

    % Select i-th row of all samples
    z_sel = Z(i,:);

    % Take lesser than or equal to operations of X and Y
    leX = all(X <= z_sel,2);
    leY = all(Y <= z_sel,2);

    % Find CDF value for FX and FY given z_sel
    FX = mean(leX);
    FY = mean(leY);

    % Update test statistic if difference is found
    D(i) = abs(FX - FY);
    
end
end