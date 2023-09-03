# wafer_detection
This model will predict whether a given wafer is faulty one or not

 sqlite3 your_database_file.db 
 .output output_file.sql
 .schema your_table_name
 .dump your_table_name
 .exit

Connects to an SQLite database (self.db_file).
Sets the output of SQLite commands to be redirected to a specific SQL file (self.output_sql_file).
Retrieves the schema of a specified table and dumps its data into the same SQL file.
Exits the SQLite shell.