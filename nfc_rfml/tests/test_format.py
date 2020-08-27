import unittest
from pathlib import Path
from preprocess.format import tags_files, partition, windows_2d, windows_3d, filter_peaks_windows

DS1 = Path("../../data/dataset/1")


class FormatTest(unittest.TestCase):
    def test_tags_files(self):
        data = [1, 3]
        result = tags_files(DS1, data)
        self.assertEqual(result, [("tag1-1.nfc", "tag1-2.nfc", "tag1-3.nfc"),
                                  ("tag3-1.nfc", "tag3-2.nfc", "tag3-3.nfc")])

    def test_divisible_partition(self):
        data = [x for x in range(20)]
        result = list(partition(data, 4))
        self.assertEqual(len(result[0]), 4)
        self.assertEqual(len(result[-1]), 4)

    # def test_not_divisible_partition(self):
    #     data = [x for x in range(21)]
    #     result = list(partition(data, 4))
    #     self.assertEqual(len(result[0]), 4)
    #     self.assertEqual(len(result[-1]), 4)

    def test_windows_2d(self):
        data = [[0 + 5j, 3.1 - 1j], [0.3 - 0.7j, 7 + 0j]]
        result = [list(x) for x in windows_2d(data)]
        self.assertEqual(result, [[0, 3.1, 5, -1],
                                  [0.3, 7, -0.7, 0]])

    def test_windows_3d(self):
        data = [[0 + 5j, 3.1 - 1j], [0.3 - 0.7j, 7 + 0j]]
        result = windows_3d(data).tolist()
        self.assertEqual(result, [[[0, 3.1], [5, -1]],
                                  [[0.3, 7], [-0.7, 0]]])


if __name__ == '__main__':
    unittest.main()
