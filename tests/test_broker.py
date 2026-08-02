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
        # 使用临时数据库，避免 paper_adapter 记录污染生产 quant.db
        # （此前该测试直接写默认 DB_PATH，残留过 4 条 reason='paper_adapter' 的脏数据）
        import tempfile, os
        import storage
        fd, tmp_db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.remove(tmp_db)  # 让 init_db 以空文件创建
        try:
            storage.init_db(tmp_db)
            a = PaperAdapter({"db_path": tmp_db})
            oid = a.place_order("000001", "BUY", 100, "limit", 10.0)
            self.assertIsInstance(oid, str)
            trades = storage.get_trades(limit=10, path=tmp_db)
            self.assertEqual(len(trades), 1)
            self.assertEqual(trades[0]["reason"], "paper_adapter")
            self.assertEqual(trades[0]["date"], "")
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            for suffix in ("-wal", "-shm"):
                p = tmp_db + suffix
                if os.path.exists(p):
                    os.remove(p)

    def test_qmt_absent_returns_false(self):
        from execution.broker.qmt import QmtAdapter
        a = QmtAdapter({})
        self.assertFalse(a.connect())  # xtquant 未安装


if __name__ == "__main__":
    unittest.main()
