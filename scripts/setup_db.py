import sqlite3
import os

DATABASE_PATH = "aases.db"

conn = sqlite3.connect(DATABASE_PATH)

cursor = conn.cursor()

cursor.execute(
"""
CREATE TABLE IF NOT EXISTS evaluation_results (
id INTEGER PRIMARY KEY AUTOINCREMENT,
question_id TEXT,
pass1_marks REAL,
pass2_marks REAL,
final_marks REAL
)
"""
)

conn.commit()

conn.close()

print("Database created successfully")
