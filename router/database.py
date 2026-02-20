import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from router.config import settings
from router.models import Base

logger = logging.getLogger(__name__)


def _ensure_sqlite_setup():
    """Ensure SQLite database path is valid and writable.
    
    This function:
    1. Creates parent directories if they don't exist
    2. Ensures the database file is not a directory
    3. Touches the file to ensure it exists
    4. Verifies write permissions
    """
    if "sqlite" not in settings.database_url.lower():
        return

    db_path = settings.database_url.replace("sqlite:///", "")
    db_path = db_path.replace("sqlite://", "")
    
    # Remove any query parameters
    if "?" in db_path:
        db_path = db_path.split("?")[0]
    
    db_file = Path(db_path).absolute()
    db_dir = db_file.parent
    
    # 1. Check if the path is already a directory (common Docker mount mistake)
    if db_file.exists() and db_file.is_dir():
        error_msg = (
            f"CRITICAL: Database path {db_file} is a directory, not a file! "
            "This often happens when mounting a non-existent file in Docker. "
            "Please delete the directory on the host and restart."
        )
        logger.error(error_msg)
        raise IsADirectoryError(error_msg)

    # 2. Create parent directory if it doesn't exist
    if not db_dir.exists():
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
        except Exception as e:
            logger.error(f"Failed to create database directory {db_dir}: {e}")
            # Continue anyway, let SQLite fail if it must

    # 3. Touch the file to ensure it exists
    try:
        if not db_file.exists():
            db_file.touch(exist_ok=True)
            logger.info(f"Created initial database file: {db_file}")
    except Exception as e:
        logger.warning(f"Could not touch database file {db_file}: {e}. "
                       "This may fail if the filesystem is read-only.")

    # 4. Check for write permissions
    if db_file.exists():
        if not os.access(db_file, os.W_OK):
            logger.warning(f"Database file {db_file} is not writable! This will likely cause errors.")
        if not os.access(db_dir, os.W_OK):
            logger.warning(f"Database directory {db_dir} is not writable! SQLite may fail to create journals.")


# Run setup logic
_ensure_sqlite_setup()

engine = create_engine(
    settings.database_url,
    connect_args={
        "check_same_thread": False,
        "timeout": 20,
    } if "sqlite" in settings.database_url else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables and run migrations."""
    try:
        # Re-run setup just in case settings changed or for explicit calls
        _ensure_sqlite_setup()
        Base.metadata.create_all(bind=engine)
        _run_migrations()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Provide extra context for SQLite errors
        if "unable to open database file" in str(e).lower():
            db_path = settings.database_url.replace("sqlite:///", "").split("?")[0]
            logger.error(f"DEBUG INFO: DB Path={db_path}, Absolute={Path(db_path).absolute()}")
            logger.error(f"DEBUG INFO: Exists={Path(db_path).exists()}, IsDir={Path(db_path).is_dir()}")
            logger.error(f"DEBUG INFO: Dir Writable={os.access(Path(db_path).parent, os.W_OK)}")
        raise


def _run_migrations() -> None:
    """Run database migrations for schema changes."""
    if "sqlite" not in settings.database_url.lower():
        return
    
    with engine.connect() as conn:
        # Check if model_profiles table exists
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_profiles'"
        ))
        if not result.fetchone():
            return
        
        # Get existing columns
        result = conn.execute(text("PRAGMA table_info(model_profiles)"))
        existing_columns = {row[1] for row in result.fetchall()}
        
        # Add adaptive_timeout_used column if missing
        if "adaptive_timeout_used" not in existing_columns:
            logger.info("Adding column: adaptive_timeout_used")
            conn.execute(text(
                "ALTER TABLE model_profiles ADD COLUMN adaptive_timeout_used FLOAT"
            ))
            conn.commit()
        
        # Add profiling_token_rate column if missing
        if "profiling_token_rate" not in existing_columns:
            logger.info("Adding column: profiling_token_rate")
            conn.execute(text(
                "ALTER TABLE model_profiles ADD COLUMN profiling_token_rate FLOAT"
            ))
            conn.commit()
        
        # Add extra_data column to model_benchmarks if missing (for ArtificialAnalysis)
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_benchmarks'"
        ))
        if result.fetchone():
            result = conn.execute(text("PRAGMA table_info(model_benchmarks)"))
            existing_bb_columns = {row[1] for row in result.fetchall()}
            if "extra_data" not in existing_bb_columns:
                logger.info("Adding column: extra_data")
                conn.execute(text(
                    "ALTER TABLE model_benchmarks ADD COLUMN extra_data JSON"
                ))
                conn.commit()
        
        logger.info("Database migrations completed")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
