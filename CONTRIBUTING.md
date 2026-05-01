# Contributing

Contributions should be proposed and discussed using the feature request or bug report issue template. New code is to be reviewed via a pull request in order to be added to develop.

## Code Style
Pull requests must adhere to style guidelines enforced by a variety of tools. Departures from the style (i.e. noqa comments) must be justified. The formatters and linters used by this repository are:

- Black formatter (88 column line default)
- Ruff linter
- isort import organizer
- Commit message requirements: Must take the form `[#XXX] Commit message`, not exceeding 72 characters.

It is recommended to install these tools in the editor of your choice and have them warn for noncompliance and apply autofixes on save.


# Tests

To run all tests, run
```
pytest --cov bsk_rl/env --cov-report term-missing tests
```

## Unit Tests

All functions, except for those that are purely Basilisk module configuration, should have unit tests in `tests/unittests`. Unit tests should be atomic and deterministic. Run
```
pytest --cov bsk_rl/env --cov-report term-missing tests/unittest
```
to unit test the `general_satellite_tasking` environment.

## Integration Tests

All code should be covered by integration test in `tests/integration`. These tests should make a gym environment and interact with it via the standard gym API to verify behaviors; as a result, these tests may be slower, flakier, and less deterministic than the unit tests, though fast and more robust tests are preferred. Run
```
pytest --cov bsk_rl/env --cov-report term-missing tests/integration
```
to integration test the `general_satellite_tasking` environment.

