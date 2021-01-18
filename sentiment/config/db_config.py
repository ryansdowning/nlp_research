import pyodbc

server = 'localhost'
database = 'TestDB'
username = 'SA'
password = None
driver = '{ODBC Driver 17 for SQL Server}'

conn = pyodbc.connect(f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}")
