#!/usr/bin/env python3
"""Plot the top-down (x, y) trajectory produced by 2Fast-2Lamaa's gp_map node.

Reads every trajectory_submap_*.csv (topometric maps) or trajectory_map.csv
(global map) file found in the given directory -- each has a
"timestamp, x, y, z, r0, r1, r2" header -- and saves a single PNG plot of the
combined path.
"""
import argparse
import csv
import glob
import os
import sys


def load_trajectory(filepath):
    poses = []
    with open(filepath, newline="") as f:
        if filepath.endswith(".csv"):
            reader = csv.reader(f, skipinitialspace=True)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) < 4:
                    continue
                t, x, y = float(row[0]), float(row[1]), float(row[2])
                poses.append((t, x, y))
        else:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                t, x, y = float(parts[0]), float(parts[1]), float(parts[2])
                poses.append((t, x, y))
    return poses


def find_trajectory_files(directory):
    files = sorted(
        glob.glob(os.path.join(directory, "trajectory_submap_*.csv")),
        key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p))) or 0),
    )
    if not files:
        map_csv = os.path.join(directory, "trajectory_map.csv")
        if os.path.isfile(map_csv):
            files = [map_csv]
            
    traj_txt = os.path.join(directory, "trajectory.txt")
    if os.path.isfile(traj_txt):
        files.append(traj_txt)
        
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Directory containing the trajectory_*.csv files")
    parser.add_argument(
        "-o", "--out", default=None, help="Path of the PNG to write (default: <output_dir>/trajectory.png)"
    )
    args = parser.parse_args()

    traj_files = find_trajectory_files(args.output_dir)
    if not traj_files:
        print(f"No trajectory_submap_*.csv or trajectory_map.csv found in {args.output_dir}", file=sys.stderr)
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    total_poses = 0
    for filepath in traj_files:
        poses = load_trajectory(filepath)
        if not poses:
            continue
        total_poses += len(poses)
        xs = [p[1] for p in poses]
        ys = [p[2] for p in poses]
        
        basename = os.path.basename(filepath)
        is_txt = basename.endswith(".txt")
        linewidth = 2.0 if is_txt else 1.5
        linestyle = "--" if is_txt else "-"
        zorder = 10 if is_txt else 1
        
        ax.plot(xs, ys, linewidth=linewidth, linestyle=linestyle, zorder=zorder, label=basename)

    if total_poses == 0:
        print("Trajectory files were found but contained no poses.", file=sys.stderr)
        return 1

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("2Fast-2Lamaa localization trajectory")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.5)
    if len(traj_files) > 1:
        ax.legend(fontsize="small", markerscale=0.5, ncol=2)

    out_path = args.out or os.path.join(args.output_dir, "trajectory.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved trajectory plot ({total_poses} poses, {len(traj_files)} file(s)) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
