"""Visualization utilities for DEM analysis and refinement curve inspection.

This module provides functions for visualizing DEM data, gradient fields,
and extracted refinement curves. Includes both simple and comprehensive
debug visualizations for quality control.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_results(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grad_mag: np.ndarray,
    curves: List[np.ndarray],
    threshold: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize DEM, gradients, and extracted refinement curves.

    Creates a 3-panel figure showing: (1) 3D DEM surface, (2) Gradient
    magnitude with contour threshold, and (3) Extracted curves overlaid on
    gradient background. Useful for quick verification of extraction results.

    Args:
        x: 2D array of x-coordinates
        y: 2D array of y-coordinates
        z: 2D array of elevation values
        grad_mag: 2D array of gradient magnitudes
        curves: List of curve arrays for visualization
        threshold: Gradient threshold value used for contour extraction
        save_path: Optional file path to save figure as image (default: None, display only)

    Returns:
        None
    """
    fig = plt.figure(figsize=(18, 6))

    # 3D surface
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(x, y, z, cmap="terrain", linewidth=0, antialiased=True)
    ax1.set_title("Interpolated DEM Surface")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Gradient magnitude
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(
        grad_mag,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="inferno",
    )
    ax2.contour(x, y, grad_mag, levels=[threshold], colors="cyan", linewidths=2)
    fig.colorbar(im, ax=ax2, label="|∇z|")
    ax2.set_title(f"Gradient Magnitude (threshold={threshold:.3f})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Curves overlay
    ax3 = fig.add_subplot(133)
    ax3.imshow(
        grad_mag,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="gray",
        alpha=0.7,
    )
    for i, curve in enumerate(curves):
        ax3.plot(curve[:, 0], curve[:, 1], linewidth=2, label=f"Curve {i}")
    ax3.set_title(f"Refinement Curves ({len(curves)} curves)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.show()


def debug_visualization(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grad_mag: np.ndarray,
    curves: List[np.ndarray],
    threshold: float,
) -> plt.Figure:
    """
    Create comprehensive debug visualization with 4-panel layout.

    Displays: (1) 3D DEM surface with curves overlaid, (2) Gradient magnitude
    heatmap with contour threshold, (3) Elevation map with curves, and
    (4) Statistics panel showing summary information. Ideal for inspecting
    extraction results before exporting to Cubit.

    Args:
        x: 2D array of x-coordinates
        y: 2D array of y-coordinates
        z: 2D array of elevation values
        grad_mag: 2D array of gradient magnitudes
        curves: List of curve arrays for visualization
        threshold: Gradient threshold value used for contour extraction

    Returns:
        matplotlib Figure object (displayed interactively)
    """
    fig = plt.figure(figsize=(16, 10))

    # Main 3D plot with surface and curves
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")

    # Plot surface with lower opacity
    ax1.plot_surface(
        x,
        y,
        z,
        cmap="terrain",
        linewidth=0,
        antialiased=True,
        alpha=0.7,
        edgecolor="none",
    )

    # Plot curves on top of surface - find z values for curve points
    for i, curve in enumerate(curves):
        # Get z values for curve points (approximate)
        z_vals = []
        for pt in curve:
            # Find nearest point in grid
            dist = np.sqrt((x - pt[0]) ** 2 + (y - pt[1]) ** 2)
            nearest_idx = np.unravel_index(np.argmin(dist), dist.shape)
            z_vals.append(z[nearest_idx])
        z_vals = np.array(z_vals)

        # Plot curve in 3D
        ax1.plot(
            curve[:, 0],
            curve[:, 1],
            z_vals,
            "r-",
            linewidth=3,
            label=f"Curve {i}" if i < 3 else "",
        )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D DEM with Refinement Curves", fontsize=12, fontweight="bold")
    if len(curves) <= 3:
        ax1.legend()

    # 2D gradient map with curves
    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(
        grad_mag,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="inferno",
        interpolation="bilinear",
    )
    ax2.contour(x, y, grad_mag, levels=[threshold], colors="cyan", linewidths=2)

    # Plot the simplified curves in azure/cyan color
    for i, curve in enumerate(curves):
        ax2.plot(curve[:, 0], curve[:, 1], "c-", linewidth=2, alpha=0.8)

    plt.colorbar(im, ax=ax2, label="|∇z|")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(
        f"Gradient with Curves (threshold={threshold:.6f})",
        fontsize=12,
        fontweight="bold",
    )

    # Elevation map with curves overlay
    ax3 = fig.add_subplot(2, 2, 3)
    im_elev = ax3.imshow(
        z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="terrain",
        interpolation="bilinear",
    )

    for i, curve in enumerate(curves):
        ax3.plot(curve[:, 0], curve[:, 1], "r-", linewidth=2, alpha=0.8)

    plt.colorbar(im_elev, ax=ax3, label="Elevation (m)")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title(
        "Elevation Map with Refinement Curves", fontsize=12, fontweight="bold"
    )

    # Statistics panel
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")

    stats_text = f"""
    REFINEMENT CURVES STATISTICS
    {"=" * 40}
    
    Number of curves: {len(curves)}
    Gradient threshold: {threshold:.6f}
    
    Gradient Statistics:
      Min: {grad_mag.min():.6f}
      Max: {grad_mag.max():.6f}
      Mean: {grad_mag.mean():.6f}
      Std: {grad_mag.std():.6f}
    
    Elevation Statistics:
      Min: {z.min():.2f} m
      Max: {z.max():.2f} m
      Range: {z.max() - z.min():.2f} m
    
    Domain Extent:
      X: [{x.min():.2f}, {x.max():.2f}] m
      Y: [{y.min():.2f}, {y.max():.2f}] m
    
    Curve Details:
    """

    for i, curve in enumerate(curves):
        stats_text += f"\n    Curve {i}: {len(curve)} points"

    ax4.text(
        0.05,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()

    # Return figure for potential saving
    return fig


def visualize_enlarged_curves(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grad_mag: np.ndarray,
    original_curves: List[np.ndarray],
    enlarged_curves: List[np.ndarray],
    threshold: float,
    enlargement_percentage: float,
) -> plt.Figure:
    """
    Visualize comparison between original and enlarged refinement curves.

    Creates a 2-panel figure showing: (1) Original curves in red, (2) Enlarged
    curves in blue. Useful for verifying enlargement results before exporting
    to Cubit.

    Args:
        x: 2D array of x-coordinates
        y: 2D array of y-coordinates
        z: 2D array of elevation values
        grad_mag: 2D array of gradient magnitudes
        original_curves: List of original curve arrays
        enlarged_curves: List of enlarged curve arrays
        threshold: Gradient threshold value used for contour extraction
        enlargement_percentage: Percentage by which curves were enlarged

    Returns:
        matplotlib Figure object (displayed interactively)
    """
    fig = plt.figure(figsize=(16, 7))

    # Original curves
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(
        z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="terrain",
        interpolation="bilinear",
    )
    for i, curve in enumerate(original_curves):
        ax1.plot(curve[:, 0], curve[:, 1], "r-", linewidth=2, alpha=0.8)
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Original Refinement Curves", fontsize=12, fontweight="bold")

    # Enlarged curves
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(
        z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="terrain",
        interpolation="bilinear",
    )
    for i, curve in enumerate(enlarged_curves):
        ax2.plot(curve[:, 0], curve[:, 1], "b-", linewidth=2, alpha=0.8)
    plt.colorbar(im2, ax=ax2, label="Elevation (m)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(
        f"Enlarged Refinement Curves (+{enlargement_percentage:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()

    return fig


def visualize_shrunk_curves(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grad_mag: np.ndarray,
    original_curves: List[np.ndarray],
    shrunk_curves: List[np.ndarray],
    threshold: float,
    shrinkage_percentage: float,
) -> plt.Figure:
    """
    Visualize comparison between original and shrunk refinement curves.

    Creates a 2-panel figure showing: (1) Original curves in red, (2) Shrunk
    curves in green. Useful for verifying shrinkage results before exporting
    to Cubit.

    Args:
        x: 2D array of x-coordinates
        y: 2D array of y-coordinates
        z: 2D array of elevation values
        grad_mag: 2D array of gradient magnitudes
        original_curves: List of original curve arrays
        shrunk_curves: List of shrunk curve arrays
        threshold: Gradient threshold value used for contour extraction
        shrinkage_percentage: Percentage by which curves were shrunk

    Returns:
        matplotlib Figure object (displayed interactively)
    """
    fig = plt.figure(figsize=(16, 7))

    # Original curves
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(
        z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="terrain",
        interpolation="bilinear",
    )
    for i, curve in enumerate(original_curves):
        ax1.plot(curve[:, 0], curve[:, 1], "r-", linewidth=2, alpha=0.8)
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Original Refinement Curves", fontsize=12, fontweight="bold")

    # Shrunk curves
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(
        z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="terrain",
        interpolation="bilinear",
    )
    for i, curve in enumerate(shrunk_curves):
        ax2.plot(curve[:, 0], curve[:, 1], "g-", linewidth=2, alpha=0.8)
    plt.colorbar(im2, ax=ax2, label="Elevation (m)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(
        f"Shrunk Refinement Curves (-{shrinkage_percentage:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.show()

    return fig
