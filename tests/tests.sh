
echo "*** static tests ***"
python -m pytest -v ./tests/static/

echo "*** dynamic tests ***"
python -m pytest -v ./tests/dynamic/
