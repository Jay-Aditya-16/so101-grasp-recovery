function analyze_grasp_posteriors(input_file, output_file)
% Beta(1,1) posterior from offline verified lift proxies. Not a physics check.
t = readtable(input_file, 'TextType', 'string');
action = ["low"; "center"; "high"];
attempts = zeros(3,1); successes = zeros(3,1); failures = zeros(3,1);
for i = 1:3
    valid = t.candidate == action(i) & t.status == "evaluated_lift_proxy" & ~isnan(t.label);
    attempts(i) = sum(valid);
    successes(i) = sum(t.label(valid) == 1);
    failures(i) = sum(t.label(valid) == 0);
end
alpha = successes + 1; beta = failures + 1;
posterior_mean = alpha ./ (alpha + beta);
writetable(table(action, attempts, successes, failures, alpha, beta, posterior_mean), output_file);
end
