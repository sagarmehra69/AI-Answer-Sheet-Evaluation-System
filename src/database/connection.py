import sqlite3

DATABASE_PATH = "aases.db"


def get_connection():

    return sqlite3.connect(DATABASE_PATH)
