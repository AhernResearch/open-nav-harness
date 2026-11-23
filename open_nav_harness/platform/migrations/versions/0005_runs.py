"""Kernel runs: a frozen spec and its episodes, scheduled in-process."""

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL, "
        "spec TEXT NOT NULL, contract TEXT NOT NULL, digest TEXT NOT NULL, lanes INTEGER NOT NULL, "
        "status TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0, "
        "created REAL NOT NULL, updated REAL NOT NULL)"
    )
    op.execute(
        "CREATE TABLE run_episodes (run_id TEXT NOT NULL REFERENCES runs(id), "
        "episode TEXT NOT NULL, ordinal INTEGER NOT NULL, status TEXT NOT NULL, "
        "summary TEXT, error TEXT, started REAL, updated REAL NOT NULL, "
        "PRIMARY KEY (run_id, episode))"
    )
    op.execute("CREATE INDEX run_episodes_status ON run_episodes(status)")


def downgrade():
    op.drop_table("run_episodes")
    op.drop_table("runs")
