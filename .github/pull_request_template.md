## Description
Closes #XXX

Please include a summary, motivation, and context of the changes and the related issue.

### Type of change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)

### How should this pull request be reviewed?
- [ ] By commit
- [ ] All changes at once

## How Has This Been Tested?

Please describe the tests that you ran to verify your changes.

### Passes Tests
- [ ] __Unit tests__ (General Environment only) `pytest --cov bsk_rl/env --cov-report term-missing tests/unittest`
- [ ] __Integrated tests__ (General Environment only) `pytest --cov bsk_rl/env --cov-report term-missing tests/integration`
- [ ] __Examples__ (General Environment only) `pytest tests/examples`

### Test Configuration
- Python:
- Basilisk:
- Platform: 

# Checklist:

- [ ] My code follows the style guidelines of this project (passes Black, ruff, and isort)
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] Commit messages are atomic, are in the form `Issue #XXX: Message` and have a useful message
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
