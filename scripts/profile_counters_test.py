import unittest

from profile_counters import faster_pairs, orders, parse_perf_csv, summarise_group


class CounterTests(unittest.TestCase):
    def test_perf_csv_matches_events_without_modifiers(self):
        text = (
            "# started on Thu Oct  8 2026\n\n"
            "188850,,instructions:u,326623,100.00,1.13,insn per cycle\n"
            "2639,,L1-dcache-load-misses,326623,100.00,,\n"
            "<not counted>,,branch-misses:u,0,0.00,,\n"
        )
        self.assertEqual(parse_perf_csv(text), {
            "instructions": 188850,
            "L1-dcache-load-misses": 2639,
            "branch-misses": None,
        })

    def test_orders_balance_position_and_pairwise_order(self):
        labels = ["a", "b", "c"]
        rounds = orders(labels, 6)
        for label in labels:
            positions = [order.index(label) for order in rounds]
            self.assertEqual(sorted(positions), [0, 0, 1, 1, 2, 2])
        for first in labels:
            for second in labels:
                if first != second:
                    before = sum(order.index(first) < order.index(second) for order in rounds)
                    self.assertEqual(before, 3)

    def test_two_arms_alternate(self):
        self.assertEqual(orders(["a", "b"], 4), [["a", "b"], ["b", "a"], ["b", "a"], ["a", "b"]])

    def test_group_takes_the_median_of_variant_medians(self):
        variants = [{"cycles:u": {"median": value}} for value in (120, 100, 110)]
        group = summarise_group(variants, ["cycles:u"])["cycles:u"]
        self.assertEqual((group["median"], group["min"], group["max"]), (110, 100, 120))
        self.assertAlmostEqual(group["spread"], 20 / 110)
        self.assertEqual(group["variants"], [120, 100, 110])

    def test_faster_pairs_compare_every_layout_pair(self):
        self.assertEqual(faster_pairs([100, 120], [90, 110, 130]), (3, 6))
        self.assertEqual(faster_pairs([100], [100]), (0, 1))


if __name__ == "__main__":
    unittest.main()
