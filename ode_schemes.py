"""
Сравнение аналитического решения задачи Коши

    y'' + a*y' = b,   y(x0) = y0,   y'(x0) = v0

с разностными схемами (7) и (9).
"""

import timeit
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-14


# ==========================================
# Постановка задачи
# ==========================================

@dataclass(frozen=True)
class Problem:
    a: float
    b: float
    y0: float
    v0: float
    x0: float = 0.0
    x_end: float = 5.0

    @property
    def has_damping(self) -> bool:
        return abs(self.a) > EPS

    def exact(self, x: np.ndarray) -> np.ndarray:
        """Аналитическое решение."""
        a, b, y0, v0 = self.a, self.b, self.y0, self.v0
        t = x - self.x0
        if self.has_damping:
            return y0 + (v0 - b / a) / a * (1 - np.exp(-a * t)) + b / a * t
        # a = 0:  y'' = b  ->  y = y0 + v0*t + b*t^2/2
        return y0 + v0 * t + b * t**2 / 2

    def second_derivative_at_start(self) -> float:
        """y''(x0) = b - a*y'(x0), выражено из самого ОДУ."""
        return self.b - self.a * self.v0


def make_grid(problem: Problem, h: float) -> np.ndarray:
    n = round((problem.x_end - problem.x0) / h)
    return np.linspace(problem.x0, problem.x_end, n + 1)


# ==========================================
# Разностные схемы
# Каждая схема: (problem, x) -> y на сетке x
# ==========================================

Scheme = Callable[[Problem, np.ndarray], np.ndarray]


def scheme_7(p: Problem, x: np.ndarray) -> np.ndarray:
    """Трёхточечная схема (7). Нужен стартовый y[1] из ряда Тейлора."""
    dx = np.diff(x)
    y = np.empty_like(x)
    y[0] = p.y0
    # y(h) ≈ y(0) + h*y'(0) + h^2/2*y''(0)
    y[1] = p.y0 + dx[0] * p.v0 + dx[0] ** 2 / 2 * p.second_derivative_at_start()

    for i in range(1, len(x) - 1):
        dy = y[i] - y[i - 1]
        dx_prev, dx_next = dx[i - 1], dx[i]

        if p.has_damping:
            ratio = (np.exp(-p.a * dx_next) - 1) / (1 - np.exp(p.a * dx_prev))
            c = p.b / p.a
            y[i + 1] = y[i] + (dy - c * dx_prev) * ratio + c * dx_next
        else:
            # предел формулы выше при a -> 0
            y[i + 1] = y[i] + dy * dx_next / dx_prev + p.b * dx_next * (dx_next + dx_prev) / 2

    return y


def scheme_9(p: Problem, x: np.ndarray) -> np.ndarray:
    """Одношаговая схема (9): переносим одновременно y и y'."""
    y = np.empty_like(x)
    y[0] = p.y0
    v = p.v0  # текущее приближение y'

    for i, dx in enumerate(np.diff(x), start=1):
        if p.has_damping:
            c = p.b / p.a
            y[i] = y[i - 1] + c * dx + (v - c) / p.a * (1 - np.exp(-p.a * dx))
            v = (v - c) * np.exp(-p.a * dx) + c
        else:
            y[i] = y[i - 1] + v * dx + p.b * dx**2 / 2
            v = v + p.b * dx

    return y


SCHEMES: dict[str, Scheme] = {
    "Разностная схема (7)": scheme_7,
    "Разностная схема (9)": scheme_9,
}

# Фиксированный цвет каждой схемы, одинаковый на всех рисунках
COLORS = {"Аналитическое решение": "C0", "Разностная схема (7)": "C1", "Разностная схема (9)": "C2"}


# ==========================================
# Анализ
# ==========================================

def max_error(y: np.ndarray, y_exact: np.ndarray) -> float:
    return float(np.max(np.abs(y - y_exact)))


def solve_all(problem: Problem, h: float) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x = make_grid(problem, h)
    return x, problem.exact(x), {name: s(problem, x) for name, s in SCHEMES.items()}


def run_time(scheme: Scheme, problem: Problem, x: np.ndarray, number: int = 10, repeat: int = 5) -> float:
    """Время одного расчёта схемой, с. Берём минимум из repeat серий по number запусков,
    чтобы отсечь фоновые помехи ОС."""
    return min(timeit.repeat(lambda: scheme(problem, x), number=number, repeat=repeat)) / number


def time_all(problem: Problem, h: float, **kwargs) -> dict[str, float]:
    """Время расчёта каждой схемы на сетке с шагом h, с."""
    x = make_grid(problem, h)
    return {name: run_time(s, problem, x, **kwargs) for name, s in SCHEMES.items()}


def convergence(problem: Problem, steps: list[float]) -> dict[str, list[float]]:
    """Максимальная ошибка каждой схемы для каждого шага из steps."""
    errors: dict[str, list[float]] = {name: [] for name in SCHEMES}
    for h in steps:
        x, y_exact, solutions = solve_all(problem, h)
        for name, y in solutions.items():
            errors[name].append(max_error(y, y_exact))
    return errors


def convergence_table(problem: Problem, steps: list[float]) -> None:
    """Печатает ошибку и время расчёта каждой схемы при уменьшении шага."""
    errors = convergence(problem, steps)
    times = [time_all(problem, h) for h in steps]
    print(f"{'h':>10}" + "".join(f"{name + ': ошибка':>34}{'время, мс':>12}" for name in errors))
    for k, h in enumerate(steps):
        print(f"{h:>10g}" + "".join(f"{errors[name][k]:>34.3e}{times[k][name] * 1e3:>12.4f}" for name in errors))


# ==========================================
# Визуализация
# Каждая функция принимает необязательный ax и возвращает его,
# чтобы графики можно было собирать в subplots и сохранять для статьи.
# ==========================================

def _get_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    return ax


def plot_solutions(problem: Problem, h: float, x, y_exact, solutions, ax=None):
    ax = _get_ax(ax)
    ax.plot(x, y_exact, linewidth=2, color=COLORS["Аналитическое решение"], label="Аналитическое решение")
    for name, y in solutions.items():
        ax.plot(x, y, "o--", markersize=4, color=COLORS.get(name), label=name)

    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title(rf"Решение $y'' + {problem.a}y' = {problem.b}$, шаг $h={h}$")
    ax.grid(True)
    ax.legend()
    return ax


def plot_errors(x, y_exact, solutions, ax=None):
    """|y_i - y(x_i)| вдоль сетки, логарифмическая шкала."""
    ax = _get_ax(ax)
    for name, y in solutions.items():
        # +EPS, чтобы точные нули не пропадали с логарифмической оси
        ax.semilogy(x, np.abs(y - y_exact) + EPS, "o-", markersize=3, color=COLORS.get(name), label=name)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|y_i - y(x_i)|$")
    ax.set_title("Погрешность вдоль сетки")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    return ax


def plot_convergence(problem: Problem, steps: list[float], ax=None):
    """Зависимость max-ошибки от шага h в log-log масштабе."""
    ax = _get_ax(ax)
    for name, errs in convergence(problem, steps).items():
        ax.loglog(steps, errs, "o-", color=COLORS.get(name), label=name)
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\max_i |y_i - y(x_i)|$")
    ax.set_title("Сходимость")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    return ax


# ==========================================
# Точка входа
# ==========================================

def main() -> None:
    problem = Problem(a=2, b=1, y0=1.0, v0=2.0)
    h = 0.1

    x, y_exact, solutions = solve_all(problem, h)
    times = time_all(problem, h)

    print(f"Шаг h = {h}")
    for name, y in solutions.items():
        print(f"{name}: максимальная ошибка {max_error(y, y_exact):.3e}, время {times[name] * 1e3:.4f} мс")
    print()
    convergence_table(problem, [0.2, 0.1, 0.05, 0.025])

    plot_solutions(problem, h, x, y_exact, solutions)
    plt.show()


if __name__ == "__main__":
    main()
