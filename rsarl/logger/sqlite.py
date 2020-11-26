
import sys
import sqlite3
from threading import Lock

lock = Lock()

class SqliteDB():

    def __init__(self, db_name: str="rsa-rl"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cur = self.conn.cursor()

    def create_table(self, sql: str):
        self.cur.execute(sql)
        self.conn.commit()

    def delete(self, sql: str):
        self.cur.execute(sql)
        self.conn.commit()

    def select(self, sql: str) -> list:
        tuple_list = []
        
        try:
            lock.acquire(True)
            result = self.cur.execute(sql)
            tuple_list = [e for e in result]
        finally:
            lock.release()
        
        return tuple_list

    def many_execute(self, sql: str, data: list):
        """RSADB use when update and insert *experiences*
        """
        try:
            self.cur.executemany(sql, data)
        except sqlite3.Error as e:
            print('sqlite3.Error occurred when many executing:', e.args[0])
            self.close()
            sys.exit(1)
        
        self.conn.commit()

    def insert(self, sql: str, row: tuple):
        try:
            self.cur.execute(sql, row)
        except sqlite3.Error as e:
            print('sqlite3.Error occurred when inserting:', e.args[0])
            self.close()
            sys.exit(1)

        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

