from pathlib import Path

import matplotlib.pyplot as plt

from movie_dt.tree import CLASSES, Node


def _layout(node: Node, depth: int, next_x: list[int], positions: dict[int, tuple[float, float]]) -> None:
    if node.is_leaf:
        positions[id(node)] = (float(next_x[0]), float(depth))
        next_x[0] += 1
        return
    _layout(node.left, depth + 1, next_x, positions)
    _layout(node.right, depth + 1, next_x, positions)
    left_x, _ = positions[id(node.left)]
    right_x, _ = positions[id(node.right)]
    positions[id(node)] = ((left_x + right_x) / 2, float(depth))


def _label(node: Node, feature_names: tuple[str, ...]) -> str:
    histogram = ", ".join(f"{label}:{int(count)}" for label, count in zip(CLASSES, node.class_counts) if count)
    if node.is_leaf:
        return f"rating = {node.prediction}\nn={node.n_samples}\n{histogram}"
    name = feature_names[node.feature_index]
    if len(name) > 42:
        name = name[:39] + "..."
    return f"{name} ≤ {node.threshold:.3g}\nn={node.n_samples}\n{histogram}"


def draw_tree(root: Node, feature_names: tuple[str, ...], path: Path, title: str | None = None) -> None:
    positions: dict[int, tuple[float, float]] = {}
    _layout(root, 0, [0], positions)
    xs = [x for x, _y in positions.values()]
    ys = [y for _x, y in positions.values()]
    width = max(10.0, 2.6 * (max(xs) - min(xs) + 1))
    height = max(5.0, 2.2 * (max(ys) - min(ys) + 1))

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axis_off()
    if title:
        ax.set_title(title)

    def walk(node: Node) -> None:
        x, y = positions[id(node)]
        if not node.is_leaf:
            for child, caption in ((node.left, "yes"), (node.right, "no")):
                cx, cy = positions[id(child)]
                ax.plot([x, cx], [-y, -cy], color="#555555", linewidth=1)
                ax.text((x + cx) / 2, -(y + cy) / 2, caption, ha="center", va="center", fontsize=8, color="#333333")
            walk(node.left)
            walk(node.right)
        ax.text(
            x,
            -y,
            _label(node, feature_names),
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f7fb", "edgecolor": "#2c3e50"},
        )

    walk(root)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
