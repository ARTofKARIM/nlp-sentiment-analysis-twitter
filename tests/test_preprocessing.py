"""Unit tests for text preprocessing."""

import unittest
from src.preprocessing import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.processor = TextPreprocessor()

    def test_remove_urls(self):
        text = "Check this https://example.com and http://test.com"
        result = self.processor.remove_urls(text)
        self.assertNotIn("https://", result)
        self.assertNotIn("http://", result)

    def test_remove_mentions(self):
        text = "@user1 hello @user2"
        result = self.processor.remove_mentions(text)
        self.assertNotIn("@user1", result)
        self.assertIn("hello", result)

    def test_remove_hashtags(self):
        text = "#happy day #coding"
        result = self.processor.remove_hashtags(text)
        self.assertIn("happy", result)
        self.assertNotIn("#", result)

    def test_lowercase(self):
        text = "HELLO World"
        result = self.processor.to_lowercase(text)
        self.assertEqual(result, "hello world")

    def test_full_pipeline(self):
        text = "@user Check https://t.co/abc #NLP is great! 123"
        result = self.processor.preprocess(text)
        self.assertNotIn("@user", result)
        self.assertNotIn("https", result)
        self.assertNotIn("123", result)
        self.assertTrue(len(result) > 0)

    def test_empty_input(self):
        result = self.processor.preprocess("")
        self.assertEqual(result, "")

    def test_non_string(self):
        result = self.processor.preprocess(None)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
