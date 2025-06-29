from tinydb import TinyDB, Query
import sqlite3

short_term_db = TinyDB("database/short_term.json")
conn = sqlite3.connect("database/long_term.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS memory (user_id TEXT, input TEXT, response TEXT)")
conn.commit()

def fetch(user_id):
    return short_term_db.search(Query().user_id == user_id)

def update(user_id, text, response):
    short_term_db.insert({"user_id": user_id, "input": text, "response": response})
    cursor.execute("INSERT INTO memory (user_id, input, response) VALUES (?, ?, ?)", (user_id, text, response))
    conn.commit()
