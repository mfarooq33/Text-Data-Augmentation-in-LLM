import unittest

from text_augmentation import TextAugmenter, augment_dataset, augment_text


class TextAugmentationTests(unittest.TestCase):
    def test_synonym_replacement_is_deterministic(self):
        augmenter = TextAugmenter(seed=7)
        first = augmenter.augment("This is a good and easy example.", n=2)
        second = augmenter.augment("This is a good and easy example.", n=2)
        self.assertEqual(first, second)
        self.assertTrue(all(variant != "This is a good and easy example." for variant in first))

    def test_requested_method_is_supported(self):
        variants = augment_text("a small useful test", n=2, methods=["synonym"], seed=3)
        self.assertEqual(len(variants), 2)
        self.assertEqual(len(set(variants)), 2)

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            TextAugmenter().augment("text", methods=["back_translation"])

    def test_dataset_preserves_mapping_fields(self):
        records = [
            {"text": "This is good", "label": "positive", "id": 1},
            {"text": "This is bad", "label": "negative", "id": 2},
        ]
        result = augment_dataset(
            records,
            n_per_sample=1,
            methods=["swap", "deletion"],
            seed=11,
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["label"], "positive")
        self.assertEqual(result[1]["label"], "positive")
        self.assertEqual(result[2]["label"], "negative")
        self.assertEqual(result[3]["label"], "negative")

    def test_string_records_can_be_augmented_without_original(self):
        result = augment_dataset(
            ["good small test", "bad result"],
            n_per_sample=1,
            include_original=False,
            methods=["synonym"],
            seed=2,
        )
        self.assertEqual(len(result), 2)
        self.assertNotIn("good small test", result)
        self.assertNotIn("bad result", result)


if __name__ == "__main__":
    unittest.main()
