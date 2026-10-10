import numpy as np


class Likelihood:
    def __init__(
        self,
        rel_error_floor=0.05,
        abs_error_floor=0.05,
    ):
        self.rel_error_floor = rel_error_floor
        self.abs_error_floor = abs_error_floor

    def _safe_errors(self, y_exp, yerr_exp):
        return np.maximum.reduce(
            [
                np.abs(yerr_exp),
                self.rel_error_floor * np.abs(y_exp),
                np.full_like(
                    yerr_exp,
                    self.abs_error_floor,
                    dtype=float,
                ),
            ]
        )

    def contributions(
        self,
        x_model,
        y_model,
        x_exp,
        y_exp,
        yerr_exp,
        yerr_model=None,
    ):
        x_model = np.asarray(x_model)
        y_model = np.asarray(y_model)
        x_exp = np.asarray(x_exp)
        y_exp = np.asarray(y_exp)
        yerr_exp = np.asarray(yerr_exp)

        mask = (
            np.isfinite(x_exp)
            & np.isfinite(y_exp)
            & np.isfinite(yerr_exp)
            & (yerr_exp >= 0)
            & (x_exp >= x_model.min())
            & (x_exp <= x_model.max())
        )

        x_exp = x_exp[mask]
        y_exp = y_exp[mask]
        yerr_exp = yerr_exp[mask]

        y_model_at_exp = np.interp(
            x_exp,
            x_model,
            y_model,
        )

        error = self._safe_errors(
            y_exp,
            yerr_exp,
        )

        if yerr_model is not None:
            yerr_model_at_exp = np.interp(
                x_exp,
                x_model,
                np.asarray(yerr_model),
            )

            error = np.sqrt(error**2 + yerr_model_at_exp**2)

        residual = (y_model_at_exp - y_exp) / error

        return x_exp, residual**2

    def chi2(
        self,
        x_model,
        y_model,
        x_exp,
        y_exp,
        yerr_exp,
        yerr_model=None,
    ):
        _, chi2_i = self.contributions(
            x_model,
            y_model,
            x_exp,
            y_exp,
            yerr_exp,
            yerr_model,
        )

        return np.sum(chi2_i)

    def log_likelihood(
        self,
        x_model,
        y_model,
        x_exp,
        y_exp,
        yerr_exp,
        yerr_model=None,
    ):
        return -0.5 * self.chi2(
            x_model,
            y_model,
            x_exp,
            y_exp,
            yerr_exp,
            yerr_model,
        )
