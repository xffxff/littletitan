
autoflake --in-place --quiet --remove-all-unused-imports --remove-unused-variables --recursive littletitan --exclude __init__.py
isort .
black .