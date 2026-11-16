"""Persistent interactive conversations and replayable events."""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "CREATE TABLE conversations (id TEXT PRIMARY KEY, created REAL NOT NULL, spec TEXT NOT NULL, status TEXT NOT NULL, history TEXT)"
    )
    op.execute(
        "CREATE TABLE conversation_events (seq INTEGER PRIMARY KEY AUTOINCREMENT, conversation TEXT NOT NULL REFERENCES conversations(id), kind TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    op.execute(
        "CREATE TABLE conversation_inputs (conversation TEXT NOT NULL REFERENCES conversations(id), request_id TEXT NOT NULL, content TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(conversation,request_id))"
    )


def downgrade():
    op.execute("DROP TABLE conversation_inputs")
    op.execute("DROP TABLE conversation_events")
    op.execute("DROP TABLE conversations")
