"""ART 1 as a stateful, graph-friendly PyTorch layer.

ART 1 learns a winner-take-all function on binary inputs.

The runnable experiment learns directly from a deterministic subset of the
official MNIST train split, then freezes the layer and measures its category
function on the official test split. 

Primary source: Carpenter and Grossberg (1987), "A Massively Parallel
Architecture for a Self-Organizing Neural Pattern Recognition Machine."
https://sites.bu.edu/steveg/files/2016/06/CarGro1987CVGIP.pdf
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
    r"""A Adaptive Resonance Theory layer for binary inputs.

    :meth:`~torch.nn.Module.state_dict` and :meth:`~torch.nn.Module.to`.

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
def fit(layer: ART1, patterns: Tensor, cfg: Config) -> None:
    """Present the MNIST patterns once, in order, through a compiled graph."""

    compiled_layer = torch.compile(layer, fullgraph=True)
    seen = 0
    for batch in patterns.split(cfg.batch_size):
        compiled_layer(batch.to(layer.prototype.device))
        seen += batch.shape[0]
        if seen % 1_024 == 0 or seen == patterns.shape[0]:
            print(
                f"ART1  samples={seen:>4d}/{patterns.shape[0]}  "
                f"categories={layer.num_committed}",
                flush=True,
            )


class FrozenResponses(NamedTuple):
    category: Tensor
    winning_match: Tensor


@torch.no_grad()
def frozen_responses(
    layer: ART1,
    patterns: Tensor,
    batch_size: int,
) -> FrozenResponses:
    """Evaluate without retaining the large per-category score matrices."""

    categories = []
    winning_matches = []
    for batch in patterns.split(batch_size):
        output = layer(batch.to(layer.prototype.device))
        winner = output.category.clamp_min(0).unsqueeze(-1)
        winning_match = output.match.gather(-1, winner).squeeze(-1)
        winning_match = winning_match.masked_fill(output.category < 0, torch.nan)
        categories.append(output.category.cpu())
        winning_matches.append(winning_match.cpu())
    return FrozenResponses(
        torch.cat(categories),
        torch.cat(winning_matches),
    )


def category_label_counts(
    category: Tensor,
    label: Tensor,
    num_categories: int,
) -> Tensor:
    """Post-hoc labels for interpreting an otherwise unsupervised memory."""

    accepted = category >= 0
    flat_index = category[accepted] * 10 + label[accepted]
    return torch.bincount(
        flat_index,
        minlength=num_categories * 10,
    ).reshape(num_categories, 10)


@dataclass(frozen=True)
class Metrics:
    coverage: float
    conditional_accuracy: float
    total_accuracy: float
    train_purity: float
    mean_match: float
    occupied_categories: int


def evaluate(
    responses: FrozenResponses,
    labels: Tensor,
    train_counts: Tensor,
) -> tuple[Metrics, Tensor]:
    """Measure the frozen category function using post-hoc majority labels."""

    majority_label = train_counts.argmax(dim=-1)
    accepted = responses.category >= 0
    prediction = labels.new_full(labels.shape, -1)
    prediction[accepted] = majority_label[responses.category[accepted]]
    correct = prediction == labels

    confusion_column = prediction.masked_fill(~accepted, 10)
    confusion = torch.bincount(
        labels * 11 + confusion_column,
        minlength=10 * 11,
    ).reshape(10, 11)
    labelled = train_counts.sum()
    metrics = Metrics(
        coverage=float(accepted.float().mean()),
        conditional_accuracy=float(correct[accepted].float().mean()),
        total_accuracy=float(correct.float().mean()),
        train_purity=float(train_counts.amax(dim=-1).sum() / labelled),
        mean_match=float(responses.winning_match.nanmean()),
        occupied_categories=int(
            torch.bincount(
                responses.category[accepted],
                minlength=train_counts.shape[0],
            ).count_nonzero()
        ),
    )
    return metrics, confusion


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
    layer: ART1,
    responses: FrozenResponses,
    train_counts: Tensor,
    confusion: Tensor,
    metrics: Metrics,
    cfg: Config,
    path: Path,
) -> None:
    """Show occupied logical-AND templates and the held-out function."""

    accepted = responses.category >= 0
    occupancy = torch.bincount(
        responses.category[accepted],
        minlength=layer.num_categories,
    )
    shown = occupancy.topk(min(32, int(occupancy.count_nonzero()))).indices
    majority_label = train_counts.argmax(dim=-1)

    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.8, 1.0))
    category_grid = grid[0, 0].subgridspec(4, 8)

    for plot_index, category_tensor in enumerate(shown):
        axis = figure.add_subplot(category_grid[plot_index // 8, plot_index % 8])
        category = int(category_tensor)
        draw_binary_pattern(
            axis,
            layer.prototype[category],
            cfg.image_side,
            CODES,
        )
        critical = int(layer.prototype[category].sum().item())
        axis.set_title(
            f"j={category}  digit~{int(majority_label[category])}\n"
            f"test n={int(occupancy[category])}, |t|={critical}",
            fontsize=8,
        )

    matrix_axis = figure.add_subplot(grid[0, 1])
    rate = confusion / confusion.sum(dim=-1, keepdim=True).clamp_min(1)
    matrix_axis.imshow(rate.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    for row in range(10):
        for column in range(11):
            value = float(rate[row, column])
            if value < 0.01:
                continue
            matrix_axis.text(
                column,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "#222222",
                fontsize=8,
            )
    matrix_axis.set_yticks(range(10), range(10))
    matrix_axis.set_xticks(
        range(11),
        [str(digit) for digit in range(10)] + ["reject"],
    )
    matrix_axis.set_xlabel("post-hoc majority label of frozen category")
    matrix_axis.set_ylabel("MNIST test label")
    matrix_axis.set_title(
        "Held-out category function\n"
        f"coverage={metrics.coverage:.1%}, "
        f"accuracy | accepted={metrics.conditional_accuracy:.1%}\n"
        f"total accuracy={metrics.total_accuracy:.1%}, "
        f"mean match={metrics.mean_match:.3f}",
        fontsize=11,
    )

    figure.suptitle(
        "ART 1 on binarized MNIST: choose -> test vigilance -> reset or intersect\n"
        f"rho={cfg.vigilance:.2f}, {layer.num_committed} committed, "
        f"{metrics.occupied_categories} occupied on test, "
        f"post-hoc train purity={metrics.train_purity:.1%}",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_input, train_label = binary_mnist(
        train=True,
        count=config.train_samples,
        image_side=config.image_side,
        threshold=config.pixel_threshold,
        seed=config.seed,
    )
    evaluation_input, evaluation_label = binary_mnist(
        train=False,
        count=config.evaluation_samples,
        image_side=config.image_side,
        threshold=config.pixel_threshold,
        seed=config.seed + 1,
    )
    art1 = ART1(
        in_features=train_input.shape[-1],
        num_categories=config.max_categories,
        vigilance=config.vigilance,
        choice_gain=config.choice_gain,
        device=device,
    )
    fit(art1, train_input, config)
    art1.eval()
    train_responses = frozen_responses(art1, train_input, config.batch_size)
    evaluation_responses = frozen_responses(
        art1,
        evaluation_input,
        config.batch_size,
    )
    train_counts = category_label_counts(
        train_responses.category,
        train_label,
        art1.num_categories,
    )
    metrics, confusion = evaluate(
        evaluation_responses,
        evaluation_label,
        train_counts,
    )
    render(
        art1,
        evaluation_responses,
        train_counts,
        confusion,
        metrics,
        config,
        OUT / "art1_function.png",
    )
    print(
        f"done  device={device}  categories={art1.num_committed}  "
        f"coverage={metrics.coverage:.1%}  "
        f"accuracy={metrics.total_accuracy:.1%}",
        flush=True,
    )
