
autoflake --in-place --quiet --remove-all-unused-imports --remove-unused-variables --recursive . --exclude __init__.py --exclude third_party
isort .
black .