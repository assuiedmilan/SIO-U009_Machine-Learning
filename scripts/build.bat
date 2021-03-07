pushd "%~dp0"

cd..

rmdir .eggs /q /s
rmdir build /q /s
rmdir dist /q /s

call .venv\Scripts\activate.bat

poetry build

popd