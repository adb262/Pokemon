# Project Conventions

## Python Style

- **No `getattr`**: Never use `getattr` unless absolutely necessary. Access attributes directly. If a field may not exist on all config types, add it to the relevant dataclass instead of using `getattr` with a default.
- **No lazy imports**: Never use lazy/deferred imports inside functions. All imports must be at the top of the file.
