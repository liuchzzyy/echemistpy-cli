# IO MODULE KNOWLEDGE BASE

## OVERVIEW
Manages data ingestion from various instrument formats into standardized `RawData` containers. Uses a plugin system for extensibility.

## ARCHITECTURE
`load(path)` -> `IOPluginManager` -> `BaseReader` Implementation -> `RawData`

## EXTENSION GUIDE
To add a new file format (e.g., `.new`):

1.  **Create File**: `src/echemistpy/io/plugins/NewFormat_Reader.py`
2.  **Inherit**: `from echemistpy.io.base_reader import BaseReader`
3.  **Implement**:
    - `_get_supported_extensions()`: Return `[".new"]`
    - `_load_single_file(file_path)`: Return `RawData` + `RawDataInfo`
4.  **Register**: Handled automatically by `plugin_manager` discovery.

## KEY CLASSES
- **`BaseReader`**: Template method class. Handles validation and standard metadata.
- **`IOPluginManager`**: Discovers plugins and routes `load()` calls based on file extension.
- **`RawData`**: Wrapper around `xarray.Dataset` (single file) or `xarray.DataTree` (directory).
- **`DataStandardizer`**: Helper to rename columns to standard vocabulary (e.g., `Time -> time_s`).

## CONVENTIONS
- **Metadata**: Must populate `RawDataInfo` fields (`sample_name`, `start_time`).
- **Standardization**: Use `io.column_mappings` to map instrument headers to internal names.
- **Error Handling**: Raise `ValueError` for format mismatches, not generic errors.
