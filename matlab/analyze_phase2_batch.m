function analyze_phase2_batch(inputdir, outputdir)
% Independently summarize saved traces. Cube height is simulator-only ground truth.
files = dir(fullfile(inputdir, '*.csv'));
if isempty(files), error('No rollout CSVs found'); end
if ~exist(outputdir, 'dir'), mkdir(outputdir); end
names = strings(numel(files), 1); peaks = zeros(numel(files), 1);
finalGrip = zeros(numel(files), 1); closeTime = nan(numel(files), 1);
closureGrip = nan(numel(files), 1); liftGrip = nan(numel(files), 1); transportGrip = nan(numel(files), 1);
fig = figure('Visible', 'off', 'Color', 'white');
tiledlayout(2, 1); ax1 = nexttile; hold(ax1, 'on');
ax2 = nexttile; hold(ax2, 'on');
set([ax1 ax2], 'Color', 'white', 'XColor', 'black', 'YColor', 'black', 'GridColor', [0.5 0.5 0.5]);
for i = 1:numel(files)
    data = readtable(fullfile(files(i).folder, files(i).name));
    names(i) = erase(files(i).name, '.csv');
    peaks(i) = 1000 * max(data.cube_z_m - data.cube_z_m(1));
    finalGrip(i) = data.gripper_position(end);
    first = find(data.gripper_target < 0.1, 1);
    if ~isempty(first), closeTime(i) = data.time_s(first); end
    if ismember('after_closure', data.Properties.VariableNames)
        index = find(data.after_closure, 1); if ~isempty(index), closureGrip(i) = data.gripper_position(index); end
        index = find(data.after_lift, 1); if ~isempty(index), liftGrip(i) = data.gripper_position(index); end
        index = find(data.after_transport, 1); if ~isempty(index), transportGrip(i) = data.gripper_position(index); end
    end
    plot(ax1, data.time_s, 1000*(data.cube_z_m - data.cube_z_m(1)), 'DisplayName', char(names(i)));
    plot(ax2, data.time_s, data.gripper_position, 'DisplayName', char(names(i)));
end
ylabel(ax1, 'Cube height change (mm)'); grid(ax1, 'on');
ylabel(ax2, 'Gripper joint (rad)'); xlabel(ax2, 'Simulated time (s)'); grid(ax2, 'on');
legend(ax1, 'Location', 'eastoutside', 'Interpreter', 'none', 'TextColor', 'black', 'Color', 'white');
exportgraphics(fig, fullfile(outputdir, 'phase2_rollout_comparison.png'), 'Resolution', 150);
close(fig);
summary = table(names, peaks, finalGrip, closeTime, closureGrip, liftGrip, transportGrip, 'VariableNames', ...
    {'trace','peak_lift_mm','final_gripper_rad','first_close_command_s', ...
    'recovery_after_closure_rad','recovery_after_lift_rad','recovery_after_transport_rad'});
writetable(summary, fullfile(outputdir, 'phase2_matlab_summary.csv'));
disp(summary);
end
