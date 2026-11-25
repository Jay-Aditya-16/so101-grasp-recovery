function analyze_rollout(csvfile, outputfile)
% Plot the measured gripper position, target, and cube height.
data = readtable(csvfile);
figure('Visible','off','Color','white');
tiledlayout(2,1);
nexttile;
set(gca,'Color','white','XColor','black','YColor','black');
plot(data.time_s, data.gripper_position, data.time_s, data.gripper_target);
ylabel('Gripper (rad)');
legend('Measured','Target');
grid on;
nexttile;
set(gca,'Color','white','XColor','black','YColor','black');
plot(data.time_s, 1000*(data.cube_z_m - data.cube_z_m(1)));
xlabel('Simulated time (s)');
ylabel('Cube lift (mm)');
grid on;
exportgraphics(gcf, outputfile);
fprintf('Peak cube lift: %.2f mm\n', 1000*max(data.cube_z_m - data.cube_z_m(1)));
close(gcf);
end
