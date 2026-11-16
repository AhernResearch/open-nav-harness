"""Freeze configuration with each conversation."""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE conversations ADD COLUMN configuration TEXT")


def downgrade():
    op.drop_column("conversations", "configuration")
