"""Interactive explorer for the infinite-tetration escape fractal.

For each pixel base ``c`` in the complex plane, iterate

    z_{0} = 1
    z_{n+1} = exp(z_n * log(c))

(the principal branch of ``c**z``) and color by how soon ``|z|`` diverges.
Pixels that stay finite for ``max_iter`` steps are treated as interior
(the black “lake”). There is no cheaper closed form for this coloring:
Lambert-W / Kneser tetration describe a holomorphic interpolant of one
fixed base, not per-pixel escape time.

Compute is a fused CUDA ``ElementwiseKernel`` (CuPy). One thread per
pixel computes ``log(c)`` once, then loops with per-pixel early exit.
Host transfers are integer escape (interior / hover / Iteration coloring)
and unclamped smooth bailout (default colormap).

Navigation keeps the last full-resolution image stretched to the new
view until a debounced GPU recompute finishes.

Requires NVIDIA CUDA 12.x, CuPy, NumPy, Matplotlib, and a Qt5 backend
(``MPLBACKEND=Qt5Agg``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.widgets import Button

_TETRATION_ESCAPE_KERNEL: Any = None

# Conservative skip: infinite tetration converges on the real interval
# [e^{-e}, e^{1/e}]. Only pixels essentially on that segment are skipped;
# the rest of the black set still iterates to max_iter.
_CONVERGENT_RE_MIN = math.exp(-math.e)
_CONVERGENT_RE_MAX = math.exp(1.0 / math.e)
_CONVERGENT_IM_EPS = 1e-12

ColorValueSource = Literal["iteration", "smooth"]


class EscapeField(NamedTuple):
    """Integer escape step plus unclamped continuous bailout index."""

    escape: np.ndarray
    smooth: np.ndarray


def _get_tetration_escape_kernel() -> Any:
    """Return the cached fused CUDA kernel for per-pixel escape iteration.

    Writes integer ``escape`` (step at bailout, or ``max_iter``) and
    unclamped float64 ``smooth`` for histogram coloring.

    Returns:
        CuPy ElementwiseKernel writing escape and smooth per pixel.
    """
    global _TETRATION_ESCAPE_KERNEL
    if _TETRATION_ESCAPE_KERNEL is None:
        _TETRATION_ESCAPE_KERNEL = cp.ElementwiseKernel(
            "float64 x_min, float64 x_max, float64 y_min, float64 y_max, "
            "int32 width, int32 height, int32 max_iter, float64 escape_radius_sq, "
            "float64 convergent_re_min, float64 convergent_re_max, "
            "float64 convergent_im_eps",
            "int32 escape, float64 smooth",
            r"""
            double log2_val = log(2.0);
            int col = i % width;
            int row = i / width;
            double denom_x = width > 1 ? (double)(width - 1) : 1.0;
            double denom_y = height > 1 ? (double)(height - 1) : 1.0;
            double c_re = x_min + (x_max - x_min) * (double)col / denom_x;
            double c_im = y_min + (y_max - y_min) * (double)row / denom_y;

            if (fabs(c_im) <= convergent_im_eps
                && c_re >= convergent_re_min && c_re <= convergent_re_max) {
                escape = max_iter;
                smooth = (double)max_iter;
                return;
            }

            double r = sqrt(c_re * c_re + c_im * c_im);
            if (r == 0.0) {
                escape = 0;
                smooth = 0.0;
                return;
            }
            double log_c_re = log(r);
            double log_c_im = atan2(c_im, c_re);

            double z_re = 1.0;
            double z_im = 0.0;

            for (int step = 0; step < max_iter; ++step) {
                double prod_re = z_re * log_c_re - z_im * log_c_im;
                double prod_im = z_re * log_c_im + z_im * log_c_re;
                if (prod_re > 700.0) {
                    escape = step;
                    smooth = (double)step + 1.0 - prod_re / log2_val;
                    return;
                }
                double ez = exp(prod_re);
                double sin_im;
                double cos_im;
                sincos(prod_im, &sin_im, &cos_im);
                z_re = ez * cos_im;
                z_im = ez * sin_im;

                if (!isfinite(z_re) || !isfinite(z_im)) {
                    escape = step;
                    smooth = (double)step + 0.5;
                    return;
                }
                double mag_sq = z_re * z_re + z_im * z_im;
                if (mag_sq > escape_radius_sq) {
                    escape = step;
                    double mag = sqrt(mag_sq);
                    if (mag < 2.0) {
                        mag = 2.0;
                    }
                    smooth = (double)step + 1.0 - log(log(mag)) / log2_val;
                    return;
                }
            }
            escape = max_iter;
            smooth = (double)max_iter;
            """,
            "tetration_escape",
        )
    return _TETRATION_ESCAPE_KERNEL


def compute_escape(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: int,
    height: int,
    max_iter: int = 80,
    escape_radius: float = 1e2,
) -> EscapeField:
    """Compute escape-time fields for infinite tetration ``z = c**z``.

    One CUDA launch; no per-iteration host sync. Interior pixels (no
    bailout within ``max_iter``) receive ``max_iter`` in both channels.
    Escaped pixels get the integer step plus an unclamped continuous
    index used for default coloring.

    Args:
        x_min: Minimum real coordinate of the view.
        x_max: Maximum real coordinate of the view.
        y_min: Minimum imaginary coordinate of the view.
        y_max: Maximum imaginary coordinate of the view.
        width: Image width in pixels.
        height: Image height in pixels.
        max_iter: Maximum iteration count before treating a pixel as interior.
        escape_radius: Magnitude threshold for divergence.

    Returns:
        EscapeField with int32 escape and float64 smooth arrays.
    """
    kernel = _get_tetration_escape_kernel()
    escape_radius_sq = escape_radius * escape_radius
    size = width * height
    flat_escape = cp.empty(size, dtype=cp.int32)
    flat_smooth = cp.empty(size, dtype=cp.float64)
    kernel(
        x_min,
        x_max,
        y_min,
        y_max,
        width,
        height,
        max_iter,
        escape_radius_sq,
        _CONVERGENT_RE_MIN,
        _CONVERGENT_RE_MAX,
        _CONVERGENT_IM_EPS,
        flat_escape,
        flat_smooth,
    )
    return EscapeField(
        escape=cp.asnumpy(flat_escape.reshape(height, width)),
        smooth=cp.asnumpy(flat_smooth.reshape(height, width)),
    )


def _escape_to_histogram_normalized(
    values: np.ndarray,
    max_iter: int,
    escaped_mask: np.ndarray,
    *,
    integer_bins: bool,
) -> np.ndarray:
    """Histogram-equalize escaped pixels into [0, 1] for colormap lookup.

    Rank equalization is used for the continuous (smooth) field so nearby
    bailout values get distinct colors. Discrete coloring uses a bincount
    LUT over integer iterations.

    Args:
        values: Per-pixel scalar coloring values.
        max_iter: Maximum iteration used when computing escape.
        escaped_mask: Pixels that escaped before max_iter.
        integer_bins: Use discrete iteration histogram when True.

    Returns:
        float array in [0, 1], same shape as values.
    """
    out = np.zeros(values.shape, dtype=np.float64)
    if not np.any(escaped_mask) or max_iter <= 0:
        return out

    if integer_bins:
        escape_int = values[escaped_mask].astype(np.int32)
        hist = np.bincount(escape_int.ravel(), minlength=max_iter)[:max_iter]
        total = hist.sum()
        if total <= 0:
            return out
        cdf = np.cumsum(hist.astype(np.float64)) / total
        lut = np.zeros(max_iter + 1, dtype=np.float64)
        lut[:max_iter] = cdf
        out[escaped_mask] = lut[escape_int]
        return out

    escaped_values = values[escaped_mask].ravel()
    order = np.argsort(escaped_values, kind="stable")
    ranks = np.empty_like(escaped_values, dtype=np.float64)
    if len(escaped_values) == 1:
        ranks[0] = 1.0
    else:
        ranks[order] = np.linspace(0.0, 1.0, len(escaped_values))
    out[escaped_mask] = ranks
    return out


def colorize(
    field: EscapeField,
    cmap: str = "turbo",
    max_iter: int | None = None,
    value_source: ColorValueSource = "smooth",
) -> np.ndarray:
    """Apply histogram-equalized colormap to escape-time data.

    Interior pixels (``escape >= max_iter``) are painted black. Default
    coloring uses the unclamped smooth field so nearby bailouts stay a
    gradient rather than a single band.

    Args:
        field: Integer escape and smooth bailout channels.
        cmap: Matplotlib colormap name.
        max_iter: Interior sentinel. If None, inferred as ``escape.max()``.
        value_source: Color by continuous bailout or integer iteration.

    Returns:
        RGBA float array of shape (height, width, 4).
    """
    escape = field.escape
    if max_iter is None:
        max_iter = int(escape.max())
    escaped_mask = escape < max_iter

    if value_source == "smooth":
        values = field.smooth
        integer_histogram = False
    else:
        values = escape.astype(np.float64)
        integer_histogram = True

    normalized = _escape_to_histogram_normalized(
        values,
        max_iter,
        escaped_mask,
        integer_bins=integer_histogram,
    )
    rgba = plt.colormaps[cmap](normalized)
    rgba[~escaped_mask] = (0.0, 0.0, 0.0, 1.0)
    return rgba


class TetrationExplorer:
    """Matplotlib viewer that recomputes the fractal after zoom and pan.

    The first frame is a full-resolution GPU render. Subsequent navigation
    only changes axis limits so the existing image is cropped/stretched.
    After ``recompute_idle_ms`` without further view changes, a new
    full-resolution field is computed for the current window.

    Buttons at the bottom of the figure:

    * Smooth / Iteration — histogram coloring of continuous vs integer escape
    * Colormap — cycle perceptually even matplotlib palettes

    Hover in the toolbar reports integer escape iteration once the field
    matches the current view; during debounce it reports ``recomputing…``.
    """

    _CMAP_CYCLE: tuple[str, ...] = (
        "turbo",
        "magma",
        "cividis",
        "cubehelix",
        "twilight_shifted",
        "nipy_spectral",
    )

    def __init__(
        self,
        x_min: float = -5,
        x_max: float = 5,
        y_min: float = -5,
        y_max: float = 5,
        width: int = 4096,
        height: int = 4096,
        cmap: str = "turbo",
        max_iter: int = 120,
        recompute_idle_ms: int = 125,
    ) -> None:
        """Initialize the explorer and draw the first frame.

        Args:
            x_min: Initial minimum real coordinate.
            x_max: Initial maximum real coordinate.
            y_min: Initial minimum imaginary coordinate.
            y_max: Initial maximum imaginary coordinate.
            width: Pixel width for each redraw.
            height: Pixel height for each redraw.
            cmap: Colormap name passed to colorize.
            max_iter: Maximum tetration iterations per pixel.
            recompute_idle_ms: Idle time before recomputing after navigation.
        """
        self._width = width
        self._height = height
        self._cmap = cmap
        self._max_iter = max_iter
        self._recompute_idle_ms = recompute_idle_ms
        self._value_source: ColorValueSource = "smooth"
        self._navigation_hooks_installed = False
        self._last_limits: tuple[float, float, float, float] | None = None
        self._recompute_timer: Any = None
        self._field: EscapeField | None = None
        self._last_hover_text: str | None = None
        self._base_title = "Infinite tetration escape"

        self._fig: Figure
        self._ax: Axes
        self._fig, self._ax = plt.subplots(figsize=(8, 8))
        self._ax.set_autoscale_on(False)
        self._ax.set_xlim(x_min, x_max)
        self._ax.set_ylim(y_min, y_max)
        self._ax.set_aspect("equal", adjustable="datalim")
        self._ax.use_sticky_edges = False
        self._ax.set_xlabel("Re", labelpad=2)
        self._ax.set_ylabel("Im", labelpad=2)
        self._ax.set_title(self._base_title, pad=4)
        self._ax.tick_params(pad=2)
        self._fig.subplots_adjust(left=0.04, right=0.99, bottom=0.07, top=0.97)
        self._install_color_controls()

        field = compute_escape(
            x_min, x_max, y_min, y_max, width, height, max_iter=max_iter
        )
        self._image: AxesImage = self._ax.imshow(
            np.zeros((height, width, 4)),
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
        )
        self._image.sticky_edges.x[:] = []
        self._image.sticky_edges.y[:] = []
        self._apply_framebuffer(field, (x_min, x_max, y_min, y_max))

        self._fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self._ax.callbacks.connect("ylim_changed", self._on_limits_changed)
        self._install_navigation_hooks()
        if not self._navigation_hooks_installed:
            install_cid = self._fig.canvas.mpl_connect(
                "draw_event", self._on_draw_install_hooks
            )
            self._install_hooks_cid = install_cid

    def _on_draw_install_hooks(self, _event: object) -> None:
        """Install toolbar hooks once the GUI toolbar exists."""
        self._install_navigation_hooks()
        if self._navigation_hooks_installed:
            self._fig.canvas.mpl_disconnect(self._install_hooks_cid)

    def _install_navigation_hooks(self) -> None:
        """Wrap toolbar navigation so fractal recomputes after zoom/pan."""
        if self._navigation_hooks_installed:
            return
        toolbar = self._fig.canvas.toolbar
        if toolbar is None:
            return

        for method_name in (
            "release_zoom",
            "release_pan",
            "home",
            "back",
            "forward",
        ):
            original = getattr(toolbar, method_name)
            setattr(
                toolbar,
                method_name,
                self._wrap_toolbar_method(original),
            )
        self._navigation_hooks_installed = True

    def _wrap_toolbar_method(
        self, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Return a wrapper that schedules recompute after navigation.

        Args:
            original: Toolbar method to wrap.

        Returns:
            Callable that invokes original then schedules GPU recompute.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._schedule_recompute()
            return result

        return wrapped

    def _view_limits(self) -> tuple[float, float, float, float]:
        """Return axis limits as (x_min, x_max, y_min, y_max)."""
        x0, x1 = self._ax.get_xlim()
        y0, y1 = self._ax.get_ylim()
        return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

    @staticmethod
    def _limits_close(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> bool:
        """Return True if two view rectangles are effectively the same.

        Args:
            a: First view rectangle.
            b: Second view rectangle.

        Returns:
            True when corresponding corners agree within float tolerance.
        """
        return all(
            math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12) for x, y in zip(a, b)
        )

    def _field_matches_view(self) -> bool:
        """Return True if the escape field matches the current axis limits."""
        if self._last_limits is None:
            return False
        return self._limits_close(self._view_limits(), self._last_limits)

    def _stop_recompute_timer(self) -> None:
        """Cancel a pending debounced recompute."""
        if self._recompute_timer is not None:
            self._recompute_timer.stop()
            self._recompute_timer = None

    def _schedule_recompute(self) -> None:
        """Queue a full-resolution recompute after navigation settles."""
        limits = self._view_limits()
        if self._last_limits is not None and self._limits_close(
            limits, self._last_limits
        ):
            return
        self._stop_recompute_timer()
        timer = self._fig.canvas.new_timer(interval=self._recompute_idle_ms)
        timer.single_shot = True
        timer.add_callback(self._on_recompute_timer)
        timer.start()
        self._recompute_timer = timer

    def _on_recompute_timer(self) -> None:
        """Recompute the fractal for the current view at full resolution."""
        self._recompute_timer = None
        limits = self._view_limits()
        if self._last_limits is not None and self._limits_close(
            limits, self._last_limits
        ):
            return
        self._recompute_at_limits(limits)

    def _recompute_at_limits(
        self,
        limits: tuple[float, float, float, float],
    ) -> None:
        """Run GPU escape compute and update the image for the given view.

        Args:
            limits: View rectangle (x_min, x_max, y_min, y_max).
        """
        x_min, x_max, y_min, y_max = limits
        field = compute_escape(
            x_min,
            x_max,
            y_min,
            y_max,
            self._width,
            self._height,
            max_iter=self._max_iter,
        )
        self._apply_framebuffer(field, limits)
        self._fig.canvas.draw_idle()

    def _colorize_current_field(self) -> np.ndarray:
        """Colorize the stored smooth field with current style options."""
        if self._field is None:
            raise RuntimeError("No escape field is loaded for colorization.")
        return colorize(
            self._field,
            cmap=self._cmap,
            max_iter=self._max_iter,
            value_source=self._value_source,
        )

    def _apply_framebuffer(
        self,
        field: EscapeField,
        limits: tuple[float, float, float, float],
    ) -> None:
        """Store the escape field and push a colorized frame to the image.

        Args:
            field: Integer escape and smooth bailout channels.
            limits: View rectangle matching ``field``.
        """
        self._field = field
        self._last_limits = limits
        rgba = self._colorize_current_field()
        x_min, x_max, y_min, y_max = limits
        self._image.set_data(rgba)
        self._image.set_extent([x_min, x_max, y_min, y_max])

    def _recolor_framebuffer(self) -> None:
        """Reapply colormap options without recomputing escape iterations."""
        if self._field is None:
            return
        self._image.set_data(self._colorize_current_field())
        self._fig.canvas.draw_idle()

    def _install_color_controls(self) -> None:
        """Add UI controls for smooth/iteration and colormap cycling."""
        ax_value = self._fig.add_axes([0.04, 0.01, 0.16, 0.04])
        ax_cmap = self._fig.add_axes([0.22, 0.01, 0.16, 0.04])
        self._value_button = Button(ax_value, self._value_source_label())
        self._cmap_button = Button(ax_cmap, "Colormap")
        self._value_button.on_clicked(self._on_toggle_value_source)
        self._cmap_button.on_clicked(self._on_cycle_colormap)

    def _value_source_label(self) -> str:
        """Return button label for the active value source."""
        if self._value_source == "smooth":
            return "Smooth"
        return "Iteration"

    def _on_toggle_value_source(self, _event: object) -> None:
        """Switch between smooth and integer iteration coloring."""
        if self._value_source == "smooth":
            self._value_source = "iteration"
        else:
            self._value_source = "smooth"
        self._value_button.label.set_text(self._value_source_label())
        self._recolor_framebuffer()

    def _on_cycle_colormap(self, _event: object) -> None:
        """Advance to the next colormap in the cycle."""
        if self._cmap not in self._CMAP_CYCLE:
            self._cmap = self._CMAP_CYCLE[0]
        else:
            index = self._CMAP_CYCLE.index(self._cmap)
            self._cmap = self._CMAP_CYCLE[(index + 1) % len(self._CMAP_CYCLE)]
        self._recolor_framebuffer()

    def _escape_at_data(self, x: float, y: float) -> int | None:
        """Return integer escape iteration at (x, y), or None if stale.

        Args:
            x: Real coordinate under the cursor.
            y: Imaginary coordinate under the cursor.

        Returns:
            Escape iteration, or None when the field does not match the view.
        """
        if self._field is None or self._last_limits is None:
            return None
        if not self._field_matches_view():
            return None
        x_min, x_max, y_min, y_max = self._last_limits
        if x < min(x_min, x_max) or x > max(x_min, x_max):
            return None
        if y < min(y_min, y_max) or y > max(y_min, y_max):
            return None

        width = self._field.escape.shape[1]
        height = self._field.escape.shape[0]
        col = int((x - x_min) / (x_max - x_min) * width)
        row = int((y - y_min) / (y_max - y_min) * height)
        col = int(np.clip(col, 0, width - 1))
        row = int(np.clip(row, 0, height - 1))
        return int(self._field.escape[row, col])

    def _hover_readout_text(
        self,
        x: float | None,
        y: float | None,
        escape_val: int | None,
        stale: bool,
    ) -> str:
        """Format status text for the pixel under the cursor.

        Args:
            x: Real coordinate, or None if the cursor is off-axes.
            y: Imaginary coordinate, or None if the cursor is off-axes.
            escape_val: Integer escape iteration, if available.
            stale: True while a recompute is pending for the current view.

        Returns:
            Toolbar / title string.
        """
        if x is None or y is None:
            return self._base_title
        if stale:
            return (
                f"{self._base_title} · Re={x:.5g}, Im={y:.5g} · "
                "recomputing…"
            )
        if escape_val is None:
            return self._base_title
        if escape_val >= self._max_iter:
            return (
                f"{self._base_title} · Re={x:.5g}, Im={y:.5g} · "
                f"no escape in {self._max_iter} steps"
            )
        return (
            f"{self._base_title} · Re={x:.5g}, Im={y:.5g} · "
            f"escaped at iter {escape_val}"
        )

    def _show_hover_readout(
        self,
        x: float | None,
        y: float | None,
        escape_val: int | None,
        stale: bool = False,
    ) -> None:
        """Update toolbar or title with escape info for the hovered pixel.

        Args:
            x: Real coordinate, or None if the cursor is off-axes.
            y: Imaginary coordinate, or None if the cursor is off-axes.
            escape_val: Integer escape iteration, if available.
            stale: True while a recompute is pending for the current view.
        """
        text = self._hover_readout_text(x, y, escape_val, stale)
        if text == self._last_hover_text:
            return
        self._last_hover_text = text

        toolbar = self._fig.canvas.toolbar
        if toolbar is not None and hasattr(toolbar, "set_message"):
            toolbar.set_message(text)
            return

        self._ax.set_title(text, pad=4)
        self._fig.canvas.draw_idle()

    def _on_mouse_move(self, event: object) -> None:
        """Show escape iteration for the pixel under the cursor."""
        inaxes = getattr(event, "inaxes", None)
        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if inaxes is not self._ax or xdata is None or ydata is None:
            self._show_hover_readout(None, None, None, False)
            return
        stale = not self._field_matches_view()
        escape_val = self._escape_at_data(float(xdata), float(ydata))
        self._show_hover_readout(
            float(xdata),
            float(ydata),
            escape_val,
            stale,
        )

    def _on_scroll(self, event: object) -> None:
        """Schedule recompute after scroll-wheel zoom."""
        if getattr(event, "inaxes", None) is not self._ax:
            return
        self._schedule_recompute()

    def _on_limits_changed(self, _ax: Axes) -> None:
        """Schedule recompute when axis limits change during navigation."""
        self._schedule_recompute()


def main() -> None:
    """Run the interactive tetration fractal explorer."""
    TetrationExplorer()
    plt.show()


if __name__ == "__main__":
    main()
