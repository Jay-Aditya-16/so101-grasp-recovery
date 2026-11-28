function analyze_hold_verifier(input_file, output_file)
% Recount offline stage classifications; unknown is never a correct guess.
t = readtable(input_file, 'TextType', 'string');
mode = ["gripper"; "vision"; "fusion"];
held_correct = zeros(3,1); empty_correct = zeros(3,1); errors = zeros(3,1); abstain = zeros(3,1);
for i = 1:numel(mode)
    prediction = t.(mode(i));
    known = t.truth ~= "unknown";
    held_correct(i) = sum(t.truth == "held" & prediction == "held");
    empty_correct(i) = sum(t.truth == "empty" & prediction == "empty");
    errors(i) = sum(known & prediction ~= "unknown" & prediction ~= t.truth);
    abstain(i) = sum(known & prediction == "unknown");
end
writetable(table(mode, held_correct, empty_correct, errors, abstain), output_file);
end
