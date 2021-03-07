pushd "%~dp0"

cd ..

virtualenv .venv --python=C:\Python39\python.exe
call .venv\Scripts\activate.bat
poetry update

popd