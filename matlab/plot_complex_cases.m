function plot_complex_cases(summaryCsv, outputPng)
% Plot paired lift and drawer travel from independently analyzed trace CSVs.
data = readtable(summaryCsv, 'TextType', 'string');
names = data.trace;
labels = {'Small cube', 'Green cube', 'Sphere', 'Two objects'};
keys = {'small', 'green', 'sphere', 'two-objects'};
values = zeros(4, 2);
for i = 1:4
    act = find(names == "complex-" + keys{i} + "-act-1020", 1);
    recovery = find(names == "complex-" + keys{i} + "-recovery-1020", 1);
    if isempty(act) || isempty(recovery), error('Missing pair for %s', keys{i}); end
    values(i,:) = [data.peak_lift_mm(act), data.peak_lift_mm(recovery)];
end
drawerAct = find(names == "complex-drawer-act-1020", 1);
drawerRecovery = find(names == "complex-drawer-recovery-1020", 1);
if isempty(drawerAct) || isempty(drawerRecovery), error('Missing drawer pair'); end
fig = figure('Visible', 'off', 'Color', 'white', 'Position', [100 100 1000 700]);
tiledlayout(2,1);
ax1 = nexttile; bar(ax1, values); xticks(ax1, 1:4); xticklabels(ax1, labels);
ylabel(ax1, 'Peak lift (mm)'); title(ax1, 'Pick tasks: one paired trial per case');
legend(ax1, {'ACT only', 'ACT + recovery'}, 'Location', 'northwest', 'TextColor', 'black', 'Color', 'white'); grid(ax1, 'on');
ax2 = nexttile; bar(ax2, [data.max_drawer_travel_mm(drawerAct), data.max_drawer_travel_mm(drawerRecovery)]);
yline(ax2, 60, '--', 'Success threshold', 'Color', 'black'); ylim(ax2, [0 70]); xticks(ax2, 1:2); xticklabels(ax2, {'ACT only', 'ACT + recovery'});
ylabel(ax2, 'Maximum drawer travel (mm)'); title(ax2, 'Drawer opening: one paired trial'); grid(ax2, 'on');
set([ax1 ax2], 'Color', 'white', 'XColor', 'black', 'YColor', 'black');
set([ax1.Title ax2.Title], 'Color', 'black');
exportgraphics(fig, outputPng, 'Resolution', 150); close(fig);
end
