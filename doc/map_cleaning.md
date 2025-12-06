# Map Cleaning Documentation

## Overview
The `map_cleaner` executable is a post-processing tool designed to clean LiDAR maps by removing dynamic objects (e.g., moving vehicles, pedestrians) and outliers. It uses free-space carving based on recorded scan data and the corresponding trajectory to identify and remove points that should not be part of the static environment.

## Usage

```bash
map_cleaner -d <data_folder> -v <voxel_size> [OPTIONS]
```

or

```bash
ros2 run ffastllamaa map_cleaner -d <data_folder> -v <voxel_size> [OPTIONS]
```

## Command-Line Arguments

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--data_folder` | `-d` | Path to the data folder containing the map and scan data |
| `--voxel_size` | `-v` | Voxel size for the distance field (meters) - should match the voxel size used during mapping |

### Optional Arguments

| Argument | Short | Type | Default | Description |
|----------|-------|------|---------|-------------|
| `--radius` | `-r` | double | `20.0` | Radius around each pose for free-space carving (meters) - points within this radius are evaluated |
| `--using_submaps` | `-s` | flag | disabled | Process submaps instead of a single global map |
| `--help` | `-h` | - | - | Print help message and exit |

## Input Data Requirements

The data folder must contain the following files:

### For Single Map Mode (default)
| File | Description |
|------|-------------|
| `map.ply` | The point cloud map to be cleaned |
| `trajectory_map.csv` | Trajectory file with timestamped poses (format: timestamp, x, y, z, rx, ry, rz) |
| `scans/` | Directory containing individual scan files (`.ply` format) with timestamps as filenames |

### For Submap Mode (`--using_submaps`)
| File | Description |
|------|-------------|
| `submap_0.ply`, `submap_1.ply`, ... | Individual submap point clouds to be cleaned |
| `trajectory_submap_0.csv`, `trajectory_submap_1.csv`, ... | Trajectory file for each corresponding submap |
| `scans/` | Directory containing individual scan files (`.ply` format) with timestamps as filenames |

### Scan Files
- Individual scan files must be in `.ply` format
- Filenames must be timestamps (in nanoseconds): e.g., `1638360000000000000.ply`
- Scans are automatically matched to trajectory poses based on timestamps

## Output Files

The executable generates the following files:

| File | Description |
|------|-------------|
| `map_original.ply` or `submap_X_original.ply` | Backup of the original map before cleaning |
| `map.ply` or `submap_X.ply` | Cleaned map (overwrites the input map) |


## Example Usage

### Basic Usage (Single Map)
```bash
# Clean a single map with default carving radius
map_cleaner -d /path/to/map_data -v 0.2
```

### Custom Carving Radius
```bash
# Use a larger radius for processing more distant points
map_cleaner -d /path/to/map_data -v 0.2 -r 50.0
```

### Submap-Based Cleaning
```bash
# Clean multiple submaps
map_cleaner -d /path/to/map_data -v 0.2 -s
```

### Conservative Cleaning
```bash
# Use smaller radius for more conservative cleaning (only nearby points)
map_cleaner -d /path/to/map_data -v 0.2 -r 10.0
```

