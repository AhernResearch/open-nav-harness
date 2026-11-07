from alembic import context

engine = context.config.attributes["engine"]
with engine.connect() as connection:
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()
