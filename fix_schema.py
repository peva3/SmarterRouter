#!/usr/bin/env python3
"""
Fix database schema for SmarterRouter 2.1.6.
Adds missing 'active' and 'last_seen' columns to model_profiles table.
"""
import os
import sqlite3
import sys


def find_database():
    """Find the router.db database file."""
    # Common paths
    paths = [
        'data/router.db',  # relative to current dir
        '/app/hubrouter/data/router.db',  # Docker container default
        '/data/router.db',  # alternative Docker path
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def main():
    db_path = find_database()
    if not db_path:
        print("ERROR: Could not find router.db database file.")
        print("Checked paths:")
        for path in ['data/router.db', '/app/hubrouter/data/router.db', '/data/router.db']:
            exists = os.path.exists(path)
            print(f"  {path}: {'EXISTS' if exists else 'NOT FOUND'}")
        sys.exit(1)

    print(f"Database found at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(model_profiles)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns: {columns}")

    changes = False

    # Add active column if missing
    if 'active' not in columns:
        print("Adding 'active' column...")
        try:
            cursor.execute("ALTER TABLE model_profiles ADD COLUMN active INTEGER DEFAULT 1")
            changes = True
        except sqlite3.OperationalError as e:
            print(f"  Warning: {e}")

    # Add last_seen column if missing
    if 'last_seen' not in columns:
        print("Adding 'last_seen' column...")
        try:
            cursor.execute("ALTER TABLE model_profiles ADD COLUMN last_seen DATETIME")
            changes = True
        except sqlite3.OperationalError as e:
            print(f"  Warning: {e}")

    # Update existing rows
    if changes:
        try:
            cursor.execute("UPDATE model_profiles SET active = 1 WHERE active IS NULL")
            print("Updated existing rows with active=1")
        except sqlite3.OperationalError as e:
            print(f"  Warning during update: {e}")

    conn.commit()

    # Verify
    cursor.execute("PRAGMA table_info(model_profiles)")
    new_columns = [row[1] for row in cursor.fetchall()]
    print(f"Final columns: {new_columns}")

    if 'active' in new_columns and 'last_seen' in new_columns:
        print("✓ Schema fix completed successfully.")
    else:
        print("✗ Some columns still missing.")

    conn.close()

if __name__ == "__main__":
    main()
