import pyodbc

server = "YOUR SERVER CONNECTION IP"
database = "YOUR DATABASE NAME"
username = "YOUR DATABASE USERNAME"
password = "YOUR DATABASE PASSWORD"
driver = "YOUR DATABASE DRIVER"  # "{ODBC Driver 17 for SQL Server}"

# conn = pyodbc.connect(f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}")
conn = None
