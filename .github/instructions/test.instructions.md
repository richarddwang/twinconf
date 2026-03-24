---
applyTo: "tests/**/*"
description: "Rules for writing and reviewing feature tests: structure, naming, and minimizing test cases."
---
# Constitution

The goal of tests is to verify system behaviors without knowledge of the implementation. Every aspect of tests should be user-oriented and designed for a user to review. Implementation is privileged and any aspect of source code must never be leaked in tests in any kind of way.

# Structure

## Directory and file structure

The organization of test files should reflect the user's mental model of the features like documentation, never mapping the source code structure. Each test file should represent a user-facing feature (e.g., `test_full_text_search.py`), grouped by subdirectories that represent major feature areas (e.g., `search/`, `auth/`).

## Delimitation of test functions

A test file should contain the **minimum number of test functions** needed to cover its feature. The default action is to merge scenarios into an existing test; only create a separate function when the scenarios are mutually exclusive.

Scenarios should be merged into one test function when:

- **Their setups are unifiable**: Both scenarios can coexist, and it is possible to create a single setup that is a superset of both.
    ```python
    # Bad — two tests where scenarios can coexist
    def test_dashboard_empty_widget():
        dashboard = Dashboard(widgets=["clock"])
        assert dashboard.render() == expected_empty

    def test_dashboard_populated_widget():
        dashboard = Dashboard(widgets=["weather"], data=weather_data)
        assert dashboard.render() == expected_populated

    # Good — superset setup covers both
    def test_dashboard_render():
        # "clock" has no data, "weather" has data — one setup exercises both paths
        dashboard = Dashboard(widgets=["clock", "weather"], data=weather_data)
        expected = ...
        assert dashboard.render() == expected
    ```

### Vary only feature-relevant dimensions

Do not create separate tests that differ only in an axis irrelevant to the feature under test.

```python
# Bad — tests differ only in an irrelevant axis (input format)
def test_discount_calculation_json(tmp_path): ...
def test_discount_calculation_csv(tmp_path): ...

# Good — one representative case
def test_discount_calculation(tmp_path):
    # Use whichever format is most readable; format coverage belongs in parsing tests
    ...
```

### Colocate valid and invalid inputs when errors are collected at once

Prefer system designs that collect all validation errors at once instead of raising on the first failure. This allows valid and invalid inputs to be tested in a single function.

```python
# Bad — separate tests
def test_valid_fields():
    result = form.validate({"email": "a@b.com", "age": 25})
    assert result.is_valid

def test_invalid_fields():
    with pytest.raises(ValidationError):
        form.validate({"email": "bad", "age": -1})

# Good — one test with both valid and invalid inputs
def test_field_validation():
    with pytest.raises(ValidationError) as exc_info:
        # valid fields pass silently; invalid fields are all reported at once
        form.validate({"email": "a@b.com", "age": 25, "phone": "bad", "zip": -1})
    assert str(exc_info.value) == "Invalid: 'phone', 'zip'"
```

# Test function writing guidelines

Each test function should have one setup, assert only what its name promises, and avoid redundant cases.

## Interact with the system like a user, not a developer

All stages of a test function — setup (GIVEN), action (WHEN), and assertion (THEN) — should reflect how a user interacts with the system in real-world usage:

- **Private components are off-limits**: Never access private (prefixed with `_`) attributes, methods, functions, classes, or modules.

- **Use user-facing interfaces over internal components**: Only call interfaces that a user would call in a typical end-to-end workflow. Direct access to internal components is prohibited.
    ```python
    # Good — public API only
    from myapp import Client

    # Bad — internal imports
    from myapp.http.connection_pool import ConnectionPool
    from myapp.auth.token_manager import TokenManager
    ```

- **Prefer higher-level APIs**: If an interface is primarily used by internal implementation, it is likely internal. Use the highest-level API available that covers the feature being tested, even if it is not the most direct way to trigger the behavior. For example, use `client.search(query)` instead of manually constructing a `QueryBuilder` to test search results.

## Remove redundant test cases

Remove test cases that don't add new code coverage, leaving only one representative case per code path.

```python
# Bad — multiple assertions on the same code path
assert convert("100cm") == 1.0
assert convert("200cm") == 2.0
assert convert("300cm") == 3.0

# Good — one representative case
assert convert("100cm") == 1.0
```

## GIVEN: One setup per test function

A test function must have exactly **one setup** (GIVEN). Multiple actions or assertions are fine as long as they all share that setup. A second setup after any assertion is not allowed — consider merging the setups into a single superset or splitting into separate test functions, based on the delimitation rules above.

```python
# Bad — two GIVENs
def test_user_profile():
    user = User(name="Alice")              # GIVEN 1
    assert user.name == "Alice"

    user = User(name="Bob", age=30)        # GIVEN 2 — prohibited
    assert user.age == 30

# Good — single GIVEN, multiple assertions
def test_user_profile():
    user = User(name="Alice", age=30)
    assert user.name == "Alice"
    assert user.age == 30
```

## THEN: Assert only what the test is focusing on

Do not assert behaviors unrelated to the test's purpose. If the same behavior is already tested elsewhere, the assertion is redundant and should be removed.

```python
# Bad — unrelated assertions in a discount calculation test
def test_discount_calculation():
    order = create_order(items=3, coupon="SAVE10")
    assert order.item_count == 3         # unrelated to discount
    assert order.coupon == "SAVE10"      # unrelated to discount
    assert order.discount == 10.0

# Good — only discount logic
def test_discount_calculation():
    order = create_order(items=3, coupon="SAVE10")
    assert order.discount == 10.0
```

# Readability

Tests should be easy to review at a glance by a human reader.

## Keep tests compact

A test function should be verifiable **without scrolling**. If a test is too long:

1. **Extract helper functions** to hide setup details not essential to the user story.
2. **Split into separate functions** if the test covers conceptually distinct features.

## Readable multiline strings

Always use triple-quotes for multiline strings instead of `\n` escapes. If the string is used in indentation-sensitive contexts (e.g., YAML, error messages), use `textwrap.dedent` to remove leading indentation.

```python
# Bad
config_file.write_text("database:\n  host: localhost\n  port: 5432\n")

# Good
from textwrap import dedent
config_file.write_text(dedent("""\
    database:
      host: localhost
      port: 5432
"""))
```

## Assert on full message

Assert the **complete expected message** as a `dedent`/`strip` multiline string. Do not scatter partial `assert "X" in str(...)` checks — the reviewer should see the full message at a glance.

```python
# Bad
assert "Invalid email" in str(exc_info.value)
assert "age" in str(exc_info.value)

# Good
expected = dedent("""\
    Validation failed:
      Invalid email: 'not-an-email'
      Missing required field: 'age'
""").strip()
assert str(exc_info.value) == expected
```

# Workflow

Always append these todos (in order) to your test modification plan.

[ ] Review every modified test file against these instructions, checking each rule file by file before considering the task done.
[ ] Make sure all tests are passed **via the `runTests` and `testFailure` tool**, not pytest in the terminal.