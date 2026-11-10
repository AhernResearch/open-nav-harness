"""Initial durable control plane schema."""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


SCHEMA = [
    "CREATE TABLE services (id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'registered', created REAL NOT NULL)",
    "CREATE TABLE experiments (id TEXT PRIMARY KEY, spec TEXT NOT NULL, created REAL NOT NULL)",
    "CREATE TABLE jobs (id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id), status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, pause_requested INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT)",
    "CREATE TABLE attempts (id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), episode INTEGER NOT NULL, ordinal INTEGER NOT NULL, status TEXT NOT NULL, output TEXT NOT NULL, pid INTEGER, start_ticks TEXT, created REAL NOT NULL, updated REAL NOT NULL, result TEXT, error TEXT, UNIQUE(job_id,episode,ordinal))",
    "CREATE TABLE leases (resource TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES attempts(id))",
    "CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, kind TEXT NOT NULL, entity TEXT NOT NULL, payload TEXT NOT NULL)",
    "CREATE TABLE imports (id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, created REAL NOT NULL)",
]


def upgrade():
    for statement in SCHEMA:
        op.execute(statement)


def downgrade():
    for name in ("imports", "events", "leases", "attempts", "jobs", "experiments", "services"):
        op.drop_table(name)
