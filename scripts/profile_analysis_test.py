import copy
import unittest

from profile_analysis import aggregate, mine, pattern, regions


def op(name, immediate=None):
    value = {"opcode": name}
    if immediate is not None:
        value["immediates"] = immediate
    return value


def profile(ops, counts):
    return {"schema_version": 1, "op_format_version": 1, "count_semantics": "dispatch-attempts",
            "functions": [{"function_index": 0, "ops": ops, "counts": counts}]}


class SequenceAnalysisTests(unittest.TestCase):
    def test_dynamic_counts_outweigh_static_repetition(self):
        data = profile([op("LocalGet", {"index": 0}), op("I32Const", 1), op("I32Add")], [1000] * 3)
        data["functions"].append({"function_index": 1, "ops": [op("I32Const", 0)] * 20, "counts": [0] * 20})
        total, patterns = mine(data, False)
        self.assertEqual(total, 3000)
        self.assertEqual(patterns[("LocalGet", "I32Const", "I32Add")]["dispatches_removed"], 2000)
        self.assertNotIn(("I32Const", "I32Const"), patterns)

    def test_branch_entries_and_calls_break_sequences(self):
        ops = [op("I32Const", 0), op("BrIf", {"target": 4}), op("I32Const", 1), op("Drop"),
               op("I32Const", 2), op("Drop"), op("Call", {"func_idx": 9}), op("I32Const", 3), op("Drop"), op("End")]
        _, patterns = mine(profile(ops, [100, 100, 40, 40, 100, 100, 100, 100, 100, 100]), False)
        self.assertEqual(set(patterns), {("I32Const", "Drop")})
        self.assertEqual(patterns[("I32Const", "Drop")]["executions"], 240)
        self.assertEqual(list(regions([op("BrTable", {"targets": [{"pc": 2}], "default": {"pc": 4}})] + [op("Nop")] * 5)), [(1, 2), (2, 4), (4, 6)])
        self.assertEqual(list(regions([op("Label", {"end_target": 3})] + [op("Nop")] * 4)), [(1, 3), (3, 5)])

    def test_trapped_prefix_does_not_count_unreached_suffix(self):
        ops = [op("I32Const", 10), op("LocalGet", {"index": 0}), op("I32DivU"), op("I32Const", 1), op("I32Add")]
        _, patterns = mine(profile(ops, [5, 5, 5, 4, 4]), False)
        self.assertEqual(patterns[("LocalGet", "I32DivU")]["executions"], 5)
        self.assertEqual(patterns[("I32DivU", "I32Const")]["executions"], 4)
        with self.assertRaises(ValueError):
            mine(profile(ops, [5, 5, 5, 6, 6]), False)

    def test_overlapping_matches_are_not_added_as_savings(self):
        _, patterns = mine(profile([op("LocalGet", {"index": 0})] * 5, [10, 10, 9, 9, 8]), False)
        pair = patterns[("LocalGet", "LocalGet")]
        self.assertEqual(pair["executions"], 36)
        self.assertEqual(pair["non_overlapping_executions"], 19)
        self.assertEqual(pair["dispatches_removed"], 19)
        self.assertEqual(patterns[("LocalGet",) * 3]["dispatches_removed"], 18)

    def test_normalisation_preserves_aliasing_types_and_immediates(self):
        def locals_pair(a, b):
            return [op("LocalGet", {"index": a}), op("LocalGet", {"index": b})]
        self.assertEqual(pattern(locals_pair(7, 12), True), pattern(locals_pair(2, 5), True))
        self.assertNotEqual(pattern(locals_pair(7, 7), True), pattern(locals_pair(2, 5), True))
        self.assertNotEqual(pattern([op("I32Const", 0)], True), pattern([op("I32Const", 1)], True))
        self.assertNotEqual(pattern([op("I32Load", {"align": 2, "offset": 0})], True), pattern([op("I32Load", {"align": 2, "offset": 4})], True))
        self.assertNotEqual(pattern([op("I32Const", 1)], True), pattern([op("I64Const", 1)], True))

    def test_family_weighting_does_not_reward_longer_or_related_workloads(self):
        def workload(name, family, operation, iterations):
            return {"workload": name, "family": family, "capture": name,
                    "profile": profile([op("LocalGet", {"index": 0}), op("LocalGet", {"index": 1}), op(operation)], [iterations] * 3)}
        a = workload("a", "commp", "I32Add", 1000000)
        b = workload("b", "commp", "I32Add", 1)
        c = workload("c", "other", "I32Mul", 10)
        def addition(report):
            return next(row for row in report["aggregate"]["shape"]["3"] if row["pattern"][-1] == "I32Add")
        self.assertAlmostEqual(addition(aggregate([a, c]))["dispatch_fraction"], 1 / 3)
        self.assertAlmostEqual(addition(aggregate([a, b, c]))["dispatch_fraction"], 1 / 3)
        self.assertEqual(addition(aggregate([a, b, c]))["workload_coverage"], 2)
        with self.assertRaises(ValueError):
            aggregate([a, a])
        invalid = copy.deepcopy(a)
        invalid["profile"]["op_format_version"] = 2
        with self.assertRaises(ValueError):
            aggregate([invalid])


if __name__ == "__main__":
    unittest.main()
