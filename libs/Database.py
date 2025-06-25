import os
import sqlite3
import datetime
import time
from flask import jsonify

class Database:

    def __init__(self):
        create = not os.path.exists('database.db')
        self.db = sqlite3.connect('database.db')
        if create:
            self.cursor = self.db.cursor()
            self.create("CREATE TABLE settings (label TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.create("CREATE TABLE history (ip TEXT PRIMARY KEY, data TEXT NOT NULL, datex DATE NOT NULL, timex TIME NOT NULL, blocked TINYINT NOT NULl DEFAULT 0, quarantined TINYINT DEFAULT 0, attack_label TEXT NOT NULL)")
            self.create("CREATE TABLE users (email TEXT PRIMARY KEY, username TEXT NOT NULL, password TEXT NOT NULL, name TEXT NOT NULL, phone TEXT NOT NULL, department TEXT NOT NULL)")

    def create(self, query=''):
        try:
            self.cursor.execute(query)
            self.db.commit()
        except Exception as e:
            pass


    def update(self, _id, name, department, phone, email, username, password):
        if not name or not email or not department or not phone or not username:
            return 'failed'

        cursor = self.db.cursor()
        cursor.execute("SELECT * from users WHERE id=? and password=?",(_id, password, ))
        details = cursor.fetchone()
        if not details:
            return []

        cursor = self.db.cursor()
        cursor.execute("UPDATE users set name=?, department=?, phone=?, email=?, username=? where id=?",(name, department, phone, email, username, _id))
        self.db.commit()

        cursor = self.db.cursor()
        cursor.execute("SELECT * from users WHERE username=? and password=?",(username, password, ))
        details = cursor.fetchone()

        return details

    def register(self, name, department, phone, email, username, password):
        if not name or not email or not password or not username:
            return 'failed'

        details = self.login(username, password)
        if len(details['details']) > 0 :
            return 'user_exits'

        cursor = self.db.cursor()
        cursor.execute("INSERT INTO users (name, department, phone, email, username, password) VALUES (?, ?, ?, ?, ?, ?)",(name, department, phone, email, username, password))
        self.db.commit()
        return 'done'


    def login(self, username, password):
        if not username:
            return 'failed'

        if not password:
            return 'failed'

        cursor = self.db.cursor()
        cursor.execute("SELECT * from users WHERE username=? and password=?",(username, password, ))
        details = cursor.fetchone()

        if not details:
            return {'details': []}
        else:
            value = {'details':details}
        return value


    def reset_password(self, username, new_password):
        #check if user exists
        cursor = self.db.cursor()
        cursor.execute("SELECT * from users WHERE username=?",(username, ))
        details = cursor.fetchone()

        if not details:
            return 'failed'

        cursor = self.db.cursor()
        cursor.execute("UPDATE users set password=? where username=?",(new_password, username, ))
        self.db.commit()
        return 'done'

    def getAll(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT  * FROM history", ())
        fields = cursor.fetchall()
        return fields
