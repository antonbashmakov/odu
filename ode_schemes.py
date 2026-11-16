"""
Сравнение аналитического решения задачи Коши

    y'' + a*y' = b,   y(x0) = y0,   y'(x0) = v0

с разностными схемами (7) и (9).
"""

import argparse
import timeit
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable
from unicodedata import name

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
    A: float = 0.0

    @property
    def has_damping(self) -> bool:
        return abs(self.a) > EPS

    def exact(self, x: np.ndarray) -> np.ndarray:
        """Аналитическое решение."""
        a, b, y0, v0 = self.a, self.b, self.y0, self.v0
        t = x - self.x0
        if self.has_damping:
            return y0 + (v0 - b / a) / a * (1 - np.exp(-a * t)) + b / a * t
        return y0 + v0 * t + b * t**2 / 2

    def second_derivative_at_start(self) -> float:
        """y''(x0) = b - a*y'(x0), выражено из самого ОДУ."""
        return self.b - self.a * self.v0
    def a_at_start(self, h: float) -> float:
        """Первы расчет a"""
        return (self.b ** 2 * h ** 2) / 2

@dataclass(frozen=True)
class SolutionScheme(ABC):
    problem: Problem

    @abstractmethod
    def solve(self, x: np.ndarray) -> np.ndarray:
        """Каждая схема обязана это реализовать."""    

    @abstractmethod
    def name(self) -> str:
        """Каждая схема обязана это реализовать."""    

def make_grid(problem: Problem, h: float) -> np.ndarray:
    n = round((problem.x_end - problem.x0) / h)
    return np.linspace(problem.x0, problem.x_end, n + 1)

class Scheme9(SolutionScheme):
    def solve(self, x):
        """
        Одношаговая схема (9): переносим одновременно y и y'.
        Если A не задано, то возвращаемся к линейному уравнению, представленному в аналитическом решении.
        Если же A задано, то исходное уравнение становится нелинейным: y'' + A * |y'|*y' = b, и схема (9) решает его.
        """
        p = self.problem

        y = np.empty_like(x)
        y[0] = p.y0
        v = p.v0  # текущее приближение y'

        A_ = int(p.A == 0) 
        A = p.a_at_start(x[1] - x[0]) if p.A != 0 else p.a # начальное приближение A

        for i, dx in enumerate(np.diff(x), start=1):
            if p.has_damping:
                c = p.b / A
                y[i] = y[i - 1] + c * dx + (v - c) / A * (1 - np.exp(-A * dx))
                v = (v - c) * np.exp(-A * dx) + c
                """
                при A_ = 1, то есть при A не задано, то A = v * p.A, что соответствует линейному уравнению. Используется для отлаживания.
                """
                A = p.a * A_ + abs(v) * p.A 
            else:
                y[i] = y[i - 1] + v * dx + p.b * dx**2 / 2
                v = v + p.b * dx

        return y

    def name(self):
        if self.problem.A != 0:
            return "Разностная схема (9) для y'' + A*|y'|*y' = b"
        return "Разностная схема (9) для y'' + a*y' = b"

class Scheme7(SolutionScheme):
    def solve(self, x):
        """Трёхточечная схема (7). Нужен стартовый y[1] из ряда Тейлора.
        dx = np.diff(x)
        y = np.empty_like(x)
        y[0] = p.y0
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
        """
        raise NotImplementedError("Scheme7 is not implemented yet.")

    def name(self):
        if self.problem.A != 0:
            return "Разностная схема (9) для y'' + A*|y'|*y' = b"
        return "Разностная схема (9) для y'' + a*y' = b"
    
# ==========================================
# Разностные схемы
# Каждая схема: (problem, x) -> y на сетке x
# ==========================================

Scheme = Callable[[Problem, np.ndarray], np.ndarray]

#def scheme_7(p: Problem, x: np.ndarray) -> np.ndarray:


# ==========================================
# Анализ
# ==========================================

def max_error(y: np.ndarray, y_exact: np.ndarray) -> float:
    return float(np.max(np.abs(y - y_exact)))


def solve_all(problem: Problem, schemes: list[Scheme], h: float) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]:
    """Возвращает сетку x, точное решение и для каждой схемы кортеж (имя, y)."""
    x = make_grid(problem, h)
    return x, problem.exact(x), [(s.name(), s.solve(x)) for s in schemes]


def run_time(scheme: Scheme, x: np.ndarray, number: int = 10, repeat: int = 5) -> float:
    """Время одного расчёта схемой, с. Берём минимум из repeat серий по number запусков,
    чтобы отсечь фоновые помехи ОС."""
    return min(timeit.repeat(lambda: scheme.solve(x), number=number, repeat=repeat)) / number


def time_all(problem: Problem, schemes: list[Scheme], h: float, **kwargs) -> dict[str, float]:
    """Время расчёта каждой схемы на сетке с шагом h, с."""
    x = make_grid(problem, h)
    return {s.name(): run_time(s, x, **kwargs) for s in schemes}

def convergence(problem: Problem, schemes: list[Scheme], steps: list[float]) -> dict[str, list[float]]:
    """Максимальная ошибка каждой схемы для каждого шага из steps."""
    errors: dict[str, list[float]] = {s.name(): [] for s in schemes}
    for h in steps:
        x, y_exact, solutions = solve_all(problem, schemes, h)
        for name, y in solutions:
            errors[name].append(max_error(y, y_exact))
    return errors


def convergence_table(problem: Problem, schemes: list[Scheme], steps: list[float]) -> None:
    """Печатает ошибку и время расчёта каждой схемы при уменьшении шага."""
    errors = convergence(problem, schemes, steps)
    times = [time_all(problem, schemes, h) for h in steps]
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
    ax.plot(x, y_exact, linewidth=2, label="Аналитическое решение")
    for name, y in solutions:
        ax.plot(x, y, "o--", markersize=4, label=name)

    ax.set_xlabel("x")
    ax.set_ylabel("y(x)")
    ax.set_title(rf"Решение $y'' + {problem.a}y' = {problem.b}$, шаг $h={h}$")
    ax.grid(True)
    ax.legend()
    return ax


def plot_errors(x, y_exact, solutions, ax=None):
    """|y_i - y(x_i)| вдоль сетки, логарифмическая шкала."""
    ax = _get_ax(ax)
    for name, y in solutions:
        # +EPS, чтобы точные нули не пропадали с логарифмической оси
        ax.semilogy(x, np.abs(y - y_exact) + EPS, "o-", markersize=3, label=name)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|y_i - y(x_i)|$")
    ax.set_title("Погрешность вдоль сетки")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    return ax


def plot_convergence(problem: Problem, schemes: list[Scheme], steps: list[float], ax=None):
    """Зависимость max-ошибки от шага h в log-log масштабе."""
    ax = _get_ax(ax)
    for name, errs in convergence(problem, schemes, steps).items():
        print(f"{name}: {errs}")
        ax.loglog(steps, errs, "o-", label=name)
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\max_i |y_i - y(x_i)|$")
    ax.set_title("Сходимость")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    return ax


# ==========================================
# Точка входа
# ==========================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сравнение разностных схем (7) и (9) для y'' + a*y' = b")
    parser.add_argument("--a", type=float, metavar="a", default=2.0, help="коэффициент a (по умолчанию %(default)s)")
    parser.add_argument("--b", type=float, metavar="b", default=1.0, help="правая часть b (по умолчанию %(default)s)")
    parser.add_argument("--y0", type=float, metavar="y0", default=1.0, help="y(x0) (по умолчанию %(default)s)")
    parser.add_argument("--v0", type=float, metavar="v0", default=2.0, help="y'(x0) (по умолчанию %(default)s)")
    parser.add_argument("--x0", type=float, metavar="x0", default=0.0, help="левый конец отрезка (по умолчанию %(default)s)")
    parser.add_argument("--x_end", type=float, metavar="x_end", default=5.0, help="правый конец отрезка (по умолчанию %(default)s)")
    parser.add_argument("--A", type=float, metavar="A", default=0.0, help="параметр A (по умолчанию %(default)s)")
    args = parser.parse_args(argv)
    if args.x_end <= args.x0:
        parser.error("x_end должен быть больше x0")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    problem = Problem(a=args.a, b=args.b, y0=args.y0, v0=args.v0, x0=args.x0, x_end=args.x_end, A=args.A)
    h = 0.1

    SCHEMES = {Scheme9(problem)}

    x, y_exact, solutions = solve_all(problem, SCHEMES, h)
    times = time_all(problem, SCHEMES, h)

    print(f"Шаг h = {h}")
    for name, y in solutions:
        print(f"{name}: максимальная ошибка {max_error(y, y_exact):.3e}, время {times[name] * 1e3:.4f} мс")
    print()
    convergence_table(problem, SCHEMES, [0.2, 0.1, 0.05, 0.025])

    plot_solutions(problem, h, x, y_exact, solutions)
    plt.show()


if __name__ == "__main__":
    main()
