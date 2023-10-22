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

Run tests from a specific file
```
poetry run devtest [file_from_tests_dir.py]
```

Run specific test from specific file
```
poetry run devtest [file_from_tests_dir.py]::[test_fn_name]
```