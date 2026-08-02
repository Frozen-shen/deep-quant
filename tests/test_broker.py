import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from execution.broker import get_adapter, PaperAdapter


class TestPaperAdapter(unittest.TestCase):
    def test_get_adapter_default(self):
        a = get_adapter("paper")
        self.assertIsInstance(a, PaperAdapter)

    def test_connect(self):
        a = get_adapter("paper")
        self.assertTrue(a.connect())

    def test_place_order_returns_id(self):
        a = get_adapter("paper")
        oid = a.place_order("000001", "BUY", 100, "limit", 10.0)
        self.assertIsInstance(oid, str)

    def test_qmt_absent_returns_false(self):
        from execution.broker.qmt import QmtAdapter
        a = QmtAdapter({})
        self.assertFalse(a.connect())  # xtquant 未安装


if __name__ == "__main__":
    unittest.main()
