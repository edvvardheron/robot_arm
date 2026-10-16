import sqlite3

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('robot_log.db')
cursor = conn.cursor()

#create a table to store the robot's actions and sensor data
cursor.execute('''CREATE TABLE IF NOT EXISTS robot_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    predicted_object TEXT,
                    object_coordinates TEXT,
                    joint_angles TEXT,
                    confidence REAL,
                    arm_coordinates TEXT,
                    scene_context TEXT,
                    grab_success BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    ''')
conn.commit()



conn.close()