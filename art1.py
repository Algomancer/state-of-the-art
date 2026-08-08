"""
Carpenter and Grossberg (1987), "A Massively Parallel
Architecture for a Self-Organizing Neural Pattern Recognition Machine."

In training mode, ``forward`` runs
the order-sensitive ART recurrence, updates the registered category
buffers once, then returns the batch response under the new state. In evaluation
mode, ``forward`` is a read-only vectorized category function. 
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.datasets import MNIST


SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "gng" / "data"
OUT = SCRIPT_DIR / "out"

DATA = "#4c78a8"
CODES = "#f28e2b"


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    image_side: int = 14
    train_samples: int = 4_096
    evaluation_samples: int = 2_048
    batch_size: int = 128
    max_categories: int = 1_024
    vigilance: float = 0.40
    choice_gain: float = 2.0       # L > 1 in the ART 1 fast-learning rule
    pixel_threshold: float = 0.20
    seed: int = 0


# =============================================================================
# Binarized MNIST
# =============================================================================

def binary_mnist(
    *,
    train: bool,
    count: int,
    image_side: int,
    threshold: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Return a deterministic subset in ART 1's binary input domain."""

    dataset = MNIST(DATA_DIR, train=train, download=True)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(len(dataset), generator=generator)[:count]
    images = dataset.data[index].float().unsqueeze(1) / 255.0
    labels = dataset.targets[index]

    pooling = 28 // image_side
    images = F.avg_pool2d(images, kernel_size=pooling, stride=pooling)
    patterns = (images >= threshold).to(torch.get_default_dtype())
    return patterns.flatten(1), labels



def art1_choice_match(
    input: Tensor,
    prototype: Tensor,
    choice_gain: float,
) -> tuple[Tensor, Tensor]:
    """Return ART 1 choice and match for ``input (..., D)`` and ``K`` prototypes."""

    intersection = torch.minimum(input.unsqueeze(-2), prototype)  # (..., K, D)
    intersection_size = intersection.sum(dim=-1)                  # (..., K)
    choice = (
        choice_gain
        * intersection_size
        / (choice_gain - 1.0 + prototype.sum(dim=-1))
    )
    input_size = input.sum(dim=-1, keepdim=True)
    match = intersection_size / input_size.clamp_min(
        torch.finfo(input.dtype).eps,
    )
    return choice, match


class ART1Output(NamedTuple):
    """The category decision and the tensors that produced it."""

    category: Tensor       # (...) int64; -1 means no committed resonance
    choice: Tensor         # (..., K) raw bottom-up category activity
    match: Tensor          # (..., K) fraction of input confirmed by prototype
    resonance: Tensor      # (..., K) committed categories passing vigilance


# =============================================================================
# ART 1 Layer
# =============================================================================

class ART1(nn.Module):
    r"""A fast-learning Adaptive Resonance Theory layer for binary inputs.

    Args:
        in_features: Number of binary features in each input.
        num_categories: Fixed number of available F2 category nodes.
        vigilance: Minimum confirmed-input fraction required for resonance.
        choice_gain: ART 1 parameter :math:`L > 1` controlling bottom-up choice.
        device: Initial device of the prototype buffer.
        dtype: Initial floating-point dtype of the prototype buffer.

    Shape:
        - Input: :math:`(*, D)`, where :math:`D = \text{in_features}`.
        - Category: :math:`(*)`.
        - Choice, match, and resonance: :math:`(*, K)`, where
          :math:`K = \text{num_categories}`.

    Example::

        layer = ART1(196, 1024, vigilance=0.4)
        layer.train()
        training_output = layer(train_patterns)
        layer.eval()
        output = layer(test_patterns)
        predicted_category = output.category
    """

    __constants__ = [
        "in_features",
        "num_categories",
        "vigilance",
        "choice_gain",
    ]
    in_features: int
    num_categories: int
    vigilance: float
    choice_gain: float
    prototype: Tensor
    committed: Tensor

    def __init__(
        self,
        in_features: int,
        num_categories: int,
        vigilance: float = 0.7,
        choice_gain: float = 2.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_categories <= 0:
            raise ValueError(
                f"num_categories must be positive, got {num_categories}"
            )
        if not 0.0 <= vigilance <= 1.0:
            raise ValueError(f"vigilance must be in [0, 1], got {vigilance}")
        if choice_gain <= 1.0:
            raise ValueError(f"choice_gain must be greater than 1, got {choice_gain}")
        self.in_features = in_features
        self.num_categories = num_categories
        self.vigilance = vigilance
        self.choice_gain = choice_gain

        self.register_buffer(
            "prototype",
            torch.empty((num_categories, in_features), **factory_kwargs),
        )
        self.register_buffer(
            "committed",
            torch.zeros(num_categories, dtype=torch.bool, device=device),
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Restore every category to the canonical uncommitted state."""

        self.prototype.fill_(1.0)
        self.committed.zero_()

    @property
    def num_committed(self) -> int:
        """Number of categories that have learned at least one input."""

        return int(self.committed.count_nonzero().item())

    def _select(
        self,
        choice: Tensor,
        match: Tensor,
        candidates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        resonance = (match >= self.vigilance) & candidates
        category = choice.masked_fill(~resonance, -torch.inf).argmax(dim=-1)
        category = category.masked_fill(~resonance.any(dim=-1), -1)
        return category, resonance

    def _evaluate(
        self,
        input: Tensor,
        prototype: Tensor,
        committed: Tensor,
    ) -> ART1Output:
        choice, match = art1_choice_match(
            input,
            prototype,
            self.choice_gain,
        )
        category, resonance = self._select(choice, match, committed)
        return ART1Output(category, choice, match, resonance)

    def _adapt(self, input: Tensor) -> tuple[Tensor, Tensor]:
        """Return state after the exact online recurrence, without side effects."""

        samples = input.reshape(-1, self.in_features)
        category_index = torch.arange(
            self.num_categories,
            dtype=torch.long,
            device=input.device,
        )

        def continue_learning(
            step: Tensor,
            prototype: Tensor,
            committed: Tensor,
        ) -> Tensor:
            return step < samples.shape[0]

        def learn_one(
            step: Tensor,
            prototype: Tensor,
            committed: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            sample_index = step.reshape(1, 1).expand(1, self.in_features)
            sample = torch.gather(samples, 0, sample_index).squeeze(0)
            choice, match = art1_choice_match(
                sample,
                prototype,
                self.choice_gain,
            )
            resonance = match >= self.vigilance
            category = choice.masked_fill(
                ~resonance,
                -torch.inf,
            ).argmax(dim=-1)
            winner = (category_index == category) & resonance.any(dim=-1)

            # t_J <- I intersect t_J. All other prototypes pass through.
            intersection = torch.minimum(sample.unsqueeze(0), prototype)
            prototype = torch.where(winner[:, None], intersection, prototype)
            committed = committed | winner
            return step + 1, prototype, committed

        step = torch.zeros((), dtype=torch.long, device=input.device)
        _, prototype, committed = torch.while_loop(
            continue_learning,
            learn_one,
            (step, self.prototype.clone(), self.committed.clone()),
        )
        return prototype, committed

    def forward(self, input: Tensor) -> ART1Output:
        """Learn in training mode, or evaluate frozen categories in eval mode."""

        prototype = self.prototype
        committed = self.committed

        if self.training:
            # torch.while_loop is intentionally no-grad: ART state follows its
            # discrete fast-learning equation rather than gradient descent.
            with torch.no_grad():
                prototype, committed = self._adapt(input)
                self.prototype.copy_(prototype)
                self.committed.copy_(committed)

        return self._evaluate(input, prototype, committed)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"num_categories={self.num_categories}, "
            f"vigilance={self.vigilance}, "
            f"choice_gain={self.choice_gain}"
        )


# =============================================================================
# Learning and Metrics
# =============================================================================

@torch.no_grad()
def train(cfg: Config) -> tuple[Tensor, ART1]:
    """Present each noisy binary pattern once in deterministic order."""

    train_input, _, clean_prototype = noisy_shapes(
        cfg.train_samples,
        cfg.side,
        cfg.maximum_extra_pixels,
        cfg.seed,
    )
    layer = ART1(
        in_features=train_input.shape[-1],
        num_categories=cfg.max_categories,
        vigilance=cfg.vigilance,
        choice_gain=cfg.choice_gain,
    )
    seen = 0

    for batch in train_input.split(200):
        layer(batch)
        seen += batch.shape[0]
        print(
            f"ART1  samples={seen:>3d}/{cfg.train_samples}  "
            f"categories={layer.num_committed}",
            flush=True,
        )

    return clean_prototype, layer


def category_concepts(layer: ART1, concepts: Tensor) -> tuple[Tensor, Tensor]:
    """Name each committed category by its best-matching clean concept."""

    category = torch.where(layer.committed)[0]
    intersection = torch.minimum(
        concepts[:, None, :],
        layer.prototype[category][None, :, :],
    )
    match = intersection.sum(dim=-1) / concepts.sum(dim=-1, keepdim=True)
    return category, match.argmax(dim=0)


def response_confusion(
    true_label: Tensor,
    category: Tensor,
    committed: Tensor,
) -> Tensor:
    """Count frozen category outputs for each ground-truth concept."""

    column = torch.searchsorted(committed, category.clamp_min(0))
    column = column.masked_fill(category < 0, committed.numel())
    flat_index = true_label * (committed.numel() + 1) + column
    return torch.bincount(
        flat_index,
        minlength=3 * (committed.numel() + 1),
    ).reshape(3, committed.numel() + 1)


# =============================================================================
# Visualization
# =============================================================================

def draw_binary_pattern(
    axis: plt.Axes,
    pattern: Tensor,
    side: int,
    color: str,
) -> None:
    image = pattern.reshape(side, side).cpu().numpy()
    axis.imshow(
        image,
        cmap=matplotlib.colors.ListedColormap(("#f7f7f7", color)),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_xticks([])
    axis.set_yticks([])


def render(
    concepts: Tensor,
    layer: ART1,
    evaluation_label: Tensor,
    output: ART1Output,
    committed: Tensor,
    category_name: Tensor,
    cfg: Config,
    path: Path,
) -> None:
    """Show learned templates and held-out category-function responses."""

    names = ("vertical", "horizontal", "diagonal")
    categories = committed.numel()
    category_to_concept = output.category.new_full(
        (layer.num_categories,),
        -1,
    )
    category_to_concept[committed] = category_name
    accepted = output.category >= 0
    mapped = output.category.new_full(output.category.shape, -1)
    mapped[accepted] = category_to_concept[output.category[accepted]]
    accuracy = float((mapped == evaluation_label).float().mean())
    coverage = float(accepted.float().mean())
    winning_match = output.match.gather(
        -1,
        output.category.clamp_min(0).unsqueeze(-1),
    ).squeeze(-1)
    mean_match = float(winning_match[accepted].mean())
    confusion = response_confusion(
        evaluation_label,
        output.category,
        committed,
    )

    figure = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1.6, 1.0))
    concept_grid = grid[0, 0].subgridspec(1, 3)
    category_grid = grid[1, 0].subgridspec(1, categories)

    for index, name in enumerate(names):
        axis = figure.add_subplot(concept_grid[0, index])
        draw_binary_pattern(axis, concepts[index], cfg.side, DATA)
        axis.set_title(f"input concept\n{name}", fontsize=10)

    for index in range(categories):
        axis = figure.add_subplot(category_grid[0, index])
        category = int(committed[index].item())
        draw_binary_pattern(
            axis,
            layer.prototype[category],
            cfg.side,
            CODES,
        )
        critical = int(layer.prototype[category].sum().item())
        name = names[int(category_name[index].item())]
        axis.set_title(
            f"category {category} -> {name}\n{critical} critical ON pixels",
            fontsize=10,
        )

    matrix_axis = figure.add_subplot(grid[:, 1])
    matrix_axis.imshow(confusion.cpu().numpy(), cmap="Blues", aspect="auto")
    dark_text_threshold = int(confusion.max().item()) / 2
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            value = int(confusion[row, column].item())
            matrix_axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > dark_text_threshold else "#222222",
                fontsize=10,
            )
    matrix_axis.set_yticks(range(3), names)
    matrix_axis.set_xticks(
        range(categories + 1),
        [f"category {int(index.item())}" for index in committed] + ["reject"],
        rotation=30,
        ha="right",
    )
    matrix_axis.set_xlabel("frozen ART 1 output")
    matrix_axis.set_ylabel("held-out noisy input concept")
    matrix_axis.set_title(
        "The learned Boolean function\n"
        f"coverage={coverage:.1%}, accuracy={accuracy:.1%}, "
        f"mean match={mean_match:.3f}",
        fontsize=11,
    )

    figure.suptitle(
        "ART 1: choice -> template match -> reset or logical-AND update\n"
        f"rho={cfg.vigilance:.2f}, {categories} learned categories",
        fontsize=15,
    )
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    config = Config()
    clean_concept, art1 = train(config)
    evaluation_input, evaluation_label, _ = noisy_shapes(
        config.evaluation_samples,
        config.side,
        config.maximum_extra_pixels,
        config.seed + 1,
    )
    art1.eval()
    evaluation_output = art1(evaluation_input)
    committed_category, concept = category_concepts(art1, clean_concept)
    render(
        clean_concept,
        art1,
        evaluation_label,
        evaluation_output,
        committed_category,
        concept,
        config,
        OUT / "art1_function.png",
    )
    print(f"done  categories={art1.num_committed}", flush=True)
