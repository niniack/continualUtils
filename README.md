### Bump version

Bump up version
```
git tag [-vXYZ] && poetry dynamic-versioning
```

Bump up commit in files
```
poetry dynamic-versioning
```


### Testing 

Run all tests
```
poetry run devtest .
```

Run specific tests
```
poetry run devtest [file_from_tests_dir.py]
```