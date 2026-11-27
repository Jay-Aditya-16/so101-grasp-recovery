function check_transfer_thresholds(input_file, output_file)
% Check stored FK metrics against the experimental transfer thresholds.
% This does not calculate FK independently, or validate dynamics or collisions.
t = readtable(input_file);
t.matlab_valid = logical(t.in_joint_limits) & t.position_error_m <= .005 & t.downward_axis_dot >= .70;
t.measured_tilt_deg = acosd(max(-1,min(1,t.downward_axis_dot)));
writetable(t, output_file);
end
