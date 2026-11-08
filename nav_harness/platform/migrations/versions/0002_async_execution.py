"""Record selected execution bindings and forbid duplicate active episodes."""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE attempts ADD COLUMN binding TEXT")
    op.execute(
        "CREATE UNIQUE INDEX active_episode ON attempts(job_id,episode) WHERE status IN ('running','blocked')"
    )


def downgrade():
    op.execute("DROP INDEX active_episode")
    op.drop_column("attempts", "binding")
