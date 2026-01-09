# Contributing to Subsetix Kokkos

Thank you for your interest in contributing to Subsetix Kokkos! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Workflow](#development-workflow)
- [Branching Strategy](#branching-strategy)
- [Code Style](#code-style)
- [Testing Guidelines](#testing-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

Be respectful, inclusive, and collaborative. We aim to maintain a welcoming environment for all contributors.

## Development Workflow

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone git@github.com:YOUR_USERNAME/subsetix_kokkos_2.git
   cd subsetix_kokkos_2
   ```

3. Install development dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install -y ninja-build ccache g++ clang-format

   # Install pre-commit hooks (optional but recommended)
   pip install pre-commit
   pre-commit install
   ```

4. Configure and build:
   ```bash
   cmake --preset serial
   cmake --build --preset serial
   ```

### Making Changes

1. Create a new branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes and test them:
   ```bash
   # Run format checks
   pre-commit run --all-files

   # Build and test
   cmake --build --preset serial
   ctest --preset serial
   ```

3. Commit your changes with a clear message (see [Commit Messages](#commit-messages))

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Open a pull request on GitHub

## Branching Strategy

We use a **Main + Features** branching model:

- **`main`** - Protected branch containing the latest stable code
- **`feature/*`** - Feature branches for new functionality
- **`fix/*`** - Bug fix branches
- **`docs/*`** - Documentation updates
- **`refactor/*`** - Code refactoring

### Rules

- All changes must go through pull requests
- Direct commits to `main` are not allowed
- Pull requests must pass CI checks before merging
- At least one approval is required for merging

## Code Style

### C++ Code Style

We follow modern C++20 best practices:

- **Indentation**: 2 spaces (no tabs)
- **Line width**: Max 100 characters
- **Naming conventions**:
  - Types (`class`, `struct`, `enum`): `CamelCase`
  - Functions and variables: `camelBack`
  - Constants: `UPPER_CASE`
  - Private members: `camelBack` (no prefix)
  - Macros: `UPPER_CASE`

- **Formatting**: Use `.clang-format` configuration provided in the repository

### Formatting Your Code

Before committing, run:

```bash
# Format all files
find . -name '*.hpp' -o -name '*.cpp' | xargs clang-format -i

# Or use pre-commit (recommended)
pre-commit run --all-files
```

### Code Quality

- Write clear, self-documenting code
- Add comments for complex algorithms
- Prefer `const` and `constexpr` where applicable
- Use `auto` sparingly and only when the type is obvious
- Avoid raw pointers; use smart pointers or Kokkos views

## Testing Guidelines

### Writing Tests

1. Place test files in `tests/` directory
2. Name test files as `*_test.cpp`
3. Use GoogleTest assertions (`EXPECT_*`, `ASSERT_*`)

#### Important: CUDA Compatibility

When writing tests that use Kokkos parallel constructs, **do not** use `KOKKOS_LAMBDA` directly inside `TEST()` macros. This causes compilation errors with CUDA.

**Incorrect:**
```cpp
TEST(MyTest, BadExample) {
  Kokkos::parallel_reduce(n,
    KOKKOS_LAMBDA(int i, int& sum) {
      sum += i;
    }, result);
}
```

**Correct:**
```cpp
// Helper function outside TEST()
int compute_sum(int n) {
  int result = 0;
  Kokkos::parallel_reduce(n,
    KOKKOS_LAMBDA(int i, int& sum) {
      sum += i;
    }, result);
  return result;
}

TEST(MyTest, GoodExample) {
  EXPECT_EQ(compute_sum(100), 4950);
}
```

### Running Tests

```bash
# Run all tests for a preset
ctest --preset serial

# Run specific test
./build-serial/tests/subsetix_test_main --gtest_filter=MyTest.GoodExample

# Run with verbose output
ctest --preset serial --verbose
```

### Benchmarking

Run benchmarks to measure performance:

```bash
# Run all benchmarks
./build-openmp/benchmarks/subsetix_benchmark_main

# Run specific benchmark
./build-openmp/benchmarks/subsetix_benchmark_main --benchmark_filter=BM_MyBenchmark
```

## Commit Messages

Write clear, descriptive commit messages:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system changes
- `ci`: CI/CD changes

### Examples

```
feat(geometry): add interval set union operation

Implement CSR-based interval set union using Kokkos parallel
scan for efficient computation.

Closes #123
```

```
fix(tests): resolve CUDA compilation errors in example tests

Extract lambda operations into helper functions to avoid
CUDA host/device function conflicts.

Fixes #45
```

## Pull Request Process

### Before Opening a PR

1. Ensure your code is formatted:
   ```bash
   pre-commit run --all-files
   ```

2. Run tests on all relevant backends:
   ```bash
   cmake --preset serial && cmake --build --preset serial && ctest --preset serial
   cmake --preset openmp && cmake --build --preset openmp && ctest --preset openmp
   ```

3. Update documentation if needed

### Opening a PR

1. Give your PR a clear title following commit message conventions
2. Fill in the PR template describing your changes
3. Link related issues
4. Request review from maintainers

### After Opening

- Address review feedback promptly
- Keep your branch up-to-date with `main`
- Ensure CI passes before requesting merge

## Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the README and existing code comments

## License

By contributing, you agree that your contributions will be licensed under the [BSD 3-Clause License](LICENSE).

---

Thank you for contributing to Subsetix Kokkos!
