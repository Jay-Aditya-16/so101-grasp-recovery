function analyze_phase3_bandit(memory_file, x, y, output_file)
% Independently recompute the Python disjoint ridge/UCB scores.
data = jsondecode(fileread(memory_file));
actions = {'center','x_plus','y_minus','x_plus_y_minus','x_minus','y_plus'};
f = [1; (x - .20) / .10; y / .18];
means = zeros(6,1); uncertainty = zeros(6,1); score = zeros(6,1);
for i = 1:6
    A = eye(3); b = zeros(3,1);
    for j = 1:numel(data.records)
        r = data.records(j);
        if strcmp(r.action, actions{i})
            z = [1; (r.observed_xy_m(1)-.20)/.10; r.observed_xy_m(2)/.18];
            A = A + z*z'; b = b + (2*double(r.reward)-1)*z;
        end
    end
    means(i) = f'*(A\b);
    uncertainty(i) = sqrt(f'*(A\f));
    score(i) = means(i) + .25*uncertainty(i);
end
writetable(table(actions', means, uncertainty, score, 'VariableNames', ...
    {'action','mean','uncertainty','score'}), output_file);
end
