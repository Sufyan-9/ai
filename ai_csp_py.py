# ============================================================
#  AI Notebook — CSP & Hill Climbing
#  Covers: Constraints, Simple Hill Climbing, Backtracking
#  Pakistan City Graph (Map Coloring CSP)
# ============================================================


# ────────────────────────────────────────────────────────────
# QUESTION 1 — Unary, Binary, Higher-Order Constraints
# ────────────────────────────────────────────────────────────

constraints_examples = [
    # (sentence, type, rule)
    ("Person's age",               "Unary",        "age >= 0"),
    ("Password length",            "Unary",        "len(password) >= 8"),
    ("Blood pressure (systolic)",  "Unary",        "60 <= BP <= 200"),
    ("Room temperature",           "Unary",        "16 <= temp <= 30"),
    ("Student exam score",         "Unary",        "0 <= score <= 100"),
    ("Flight seat number",         "Unary",        "seat in valid_seats"),
    ("Product price",              "Unary",        "price > 0"),
    ("Driver license age",         "Unary",        "age >= 18"),
    ("Meeting start & end time",   "Binary",       "end_time > start_time"),
    ("Husband & wife age",         "Binary",       "abs(h_age - w_age) <= 20"),
    ("Adjacent map colors",        "Binary",       "color(A) != color(B) if adjacent"),
    ("Employee & manager salary",  "Binary",       "salary(emp) < salary(manager)"),
    ("Flight departure & arrival", "Binary",       "arrival > departure + duration"),
    ("Task A & Task B schedule",   "Binary",       "start(B) >= end(A)"),
    ("Two parallel courses",       "Binary",       "time_slot(C1) != time_slot(C2)"),
    ("Loan, income, term (bank)",  "Higher-Order", "monthly_payment(loan,rate,term) <= 0.4*income"),
    ("Calories, protein, fat",     "Higher-Order", "cal<=2000 and protein>=50 and fat<=65"),
    ("Staff, shifts, tasks",       "Higher-Order", "sum(staff[shift]) >= min_required[shift]"),
    ("Supplier, warehouse, demand","Higher-Order", "sum(supply) >= sum(demand)"),
    ("Items, weight, volume, cost","Higher-Order", "sum(weight)<=W and sum(vol)<=V"),
]

print("=" * 60)
print("Q1 — Constraint Types")
print("=" * 60)
for i, (sentence, ctype, rule) in enumerate(constraints_examples, 1):
    print(f"{i:>2}. [{ctype:>12}]  {sentence}")
    print(f"      Rule: {rule}")
print()


# ────────────────────────────────────────────────────────────
# QUESTION 2 — Simple Hill Climbing
# ────────────────────────────────────────────────────────────

# ── Cell 1: Objective function ──────────────────────────────
def objective(x):
    """
    f(x) = -(x-3)^2 + 9
    Maximum at x=3, f(3)=9.
    Hill climbing MAXIMISES this function.
    """
    return -(x - 3) ** 2 + 9


# ── Cell 2: Generate neighbours ─────────────────────────────
def get_neighbours(x, step=0.5):
    """
    Return left and right neighbours at distance 'step'.
    parameter x    : current solution
    parameter step : neighbourhood distance (default 0.5)
    """
    return [x - step, x + step]


# ── Cell 3: Simple Hill Climbing loop ───────────────────────
def simple_hill_climbing(start, step=0.5, max_iter=100):
    """
    Simple Hill Climbing algorithm.

    Parameters
    ----------
    start    : initial x value (starting state)
    step     : step size for generating neighbours (default 0.5)
    max_iter : maximum iterations before forced stop (default 100)

    Returns
    -------
    current  : optimal x found
    f_val    : objective value at optimal x
    history  : list of (x, f(x)) visited
    """
    current = start                                 # initial state
    history = [(current, objective(current))]       # track path

    for i in range(max_iter):
        neighbours = get_neighbours(current, step)  # generate neighbours

        # pick the best neighbour
        best_nb = max(neighbours, key=objective)

        if objective(best_nb) > objective(current):
            current = best_nb                       # move uphill
            history.append((current, objective(current)))
        else:
            break                                   # local optimum reached

    return current, objective(current), history


# ── Cell 4: Run ─────────────────────────────────────────────
print("=" * 60)
print("Q2 — Simple Hill Climbing")
print("=" * 60)

result_x, result_f, hist = simple_hill_climbing(start=0)

print(f"Start x      = 0")
print(f"Optimal x    = {result_x}")
print(f"Optimal f(x) = {result_f}")
print(f"Steps taken  = {len(hist) - 1}")
print("\nPath:")
for step_num, (x, fx) in enumerate(hist):
    print(f"  Step {step_num}: x={x:.2f}  f(x)={fx:.3f}")
print()


# ── Cell 5: Plot (requires matplotlib) ──────────────────────
try:
    import matplotlib.pyplot as plt
    import numpy as np

    xs_plot = np.linspace(-6, 9, 300)
    ys_plot = [objective(x) for x in xs_plot]

    path_x = [h[0] for h in hist]
    path_y = [h[1] for h in hist]

    plt.figure(figsize=(8, 4))
    plt.plot(xs_plot, ys_plot, color='steelblue', label='f(x)')
    plt.plot(path_x, path_y, 'o-', color='crimson', label='HC path')
    plt.scatter([path_x[-1]], [path_y[-1]], color='green', zorder=5,
                s=80, label=f'Optimum x={path_x[-1]}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Simple Hill Climbing — f(x) = -(x-3)² + 9')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hill_climbing_plot.png', dpi=120)
    plt.show()
    print("Plot saved to hill_climbing_plot.png")
except ImportError:
    print("matplotlib not installed — skipping plot.")
print()


# ────────────────────────────────────────────────────────────
# QUESTION 3 — Pakistan City Graph + Simple Backtracking
# ────────────────────────────────────────────────────────────

# Variables
cities = ['ISL', 'ABT', 'LHR', 'KHI', 'GB']

city_names = {
    'ISL': 'Islamabad',
    'ABT': 'Abbottabad',
    'LHR': 'Lahore',
    'KHI': 'Karachi',
    'GB':  'Gilgit-Baltistan',
}

# Domain
domain = ['Green', 'Blue', 'Red', 'Yellow']

# Constraints — adjacency edges (binary: no two adjacent cities same color)
edges = [
    ('ISL', 'ABT'),   # Islamabad — Abbottabad
    ('ISL', 'LHR'),   # Islamabad — Lahore
    ('ISL', 'GB'),    # Islamabad — Gilgit-Baltistan
    ('ABT', 'GB'),    # Abbottabad — Gilgit-Baltistan
    ('ABT', 'LHR'),   # Abbottabad — Lahore
    ('LHR', 'KHI'),   # Lahore — Karachi
]


def is_consistent(city, color, assignment):
    """
    Check if assigning 'color' to 'city' violates any constraint.

    Parameters
    ----------
    city       : the city variable being assigned
    color      : the color value being tried
    assignment : current partial assignment {city: color}

    Returns True if assignment is consistent, False otherwise.
    """
    for (a, b) in edges:
        # if city is one end of an edge and the other end is assigned
        if a == city and b in assignment:
            if assignment[b] == color:
                return False          # conflict!
        if b == city and a in assignment:
            if assignment[a] == color:
                return False          # conflict!
    return True


def backtrack(assignment, depth=0):
    """
    Recursive backtracking search for map coloring.

    Parameters
    ----------
    assignment : dict of {city: color} built so far
    depth      : recursion depth (for display indentation)

    Returns complete assignment or None if unsolvable.
    """
    # Base case: all cities assigned
    if len(assignment) == len(cities):
        return assignment

    # Select next unassigned city (in order)
    city = cities[len(assignment)]
    indent = "  " * depth

    print(f"{indent}Assigning: {city_names[city]}")

    for color in domain:
        print(f"{indent}  Trying {color}...", end=" ")

        if is_consistent(city, color, assignment):
            assignment[city] = color          # assign
            print(f"OK -> {city_names[city]} = {color}")

            result = backtrack(assignment, depth + 1)
            if result is not None:
                return result

            # Backtrack
            print(f"{indent}  Backtracking from {city_names[city]} = {color}")
            del assignment[city]
        else:
            print("CONFLICT")

    return None   # trigger backtrack in caller


print("=" * 60)
print("Q3 — Pakistan City Graph: Backtracking CSP")
print("=" * 60)
print(f"Variables : {[city_names[c] for c in cities]}")
print(f"Domain    : {domain}")
print(f"Edges     : {[(city_names[a], city_names[b]) for a,b in edges]}")
print()
print("Running backtracking...\n")

solution = backtrack({})

print()
print("=" * 60)
if solution:
    print("SOLUTION FOUND:")
    for city, color in solution.items():
        print(f"  {city_names[city]:<20} = {color}")
else:
    print("No solution found.")
print("=" * 60)
print()


# ────────────────────────────────────────────────────────────
# QUESTION 4 — 20 Important Questions & Answers
# ────────────────────────────────────────────────────────────

qa_list = [
    ("What is the objective function?",
     "f(x) = -(x-3)^2 + 9. It measures solution quality; HC maximises it."),

    ("What is get_neighbours()?",
     "Returns [x-step, x+step] — candidate states near the current state."),

    ("What happens when no neighbour improves the score?",
     "The loop breaks; the algorithm reports a local optimum."),

    ("What is a local vs global optimum?",
     "Local: best in neighbourhood. Global: best overall. HC can get stuck locally."),

    ("Why does step size matter?",
     "Too large skips the optimum; too small causes slow convergence."),

    ("What is a CSP?",
     "Constraint Satisfaction Problem: Variables + Domains + Constraints."),

    ("What are the variables in the Pakistan graph?",
     "ISL, ABT, LHR, KHI, GB (the five cities)."),

    ("What is the domain in this CSP?",
     "['Green', 'Blue', 'Red', 'Yellow'] — four possible colors per city."),

    ("What are the constraints in the Pakistan graph?",
     "Adjacent cities must have different colors (binary constraints)."),

    ("What does is_consistent() do?",
     "Checks if assigning a color to a city conflicts with already-assigned neighbours."),

    ("Why delete assignment[city] when backtracking?",
     "To undo the failed assignment and restore state for the next value attempt."),

    ("What is arc consistency?",
     "AC-3 pre-filters domains removing values that can't satisfy constraints early."),

    ("What is the worst-case complexity of backtracking here?",
     "O(d^n) = 4^5 = 1024 combinations. Backtracking prunes most branches."),

    ("Why use map coloring as a CSP?",
     "It's a classic NP-complete problem demonstrating binary constraints cleanly."),

    ("What are unary constraints?",
     "Constraints on a single variable, e.g. age >= 18."),

    ("What are higher-order constraints?",
     "Constraints involving 3+ variables, e.g. loan repayment <= 40% of income."),

    ("How to minimise instead of maximise in HC?",
     "Change: if objective(best_nb) < objective(current): move. Descend instead of ascend."),

    ("What is random restart hill climbing?",
     "Restart from random states when stuck locally, improving global search."),

    ("How does backtracking differ from brute force?",
     "Brute force tries all d^n combos. Backtracking prunes early on constraint failure."),

    ("What is forward checking in CSP?",
     "After assigning a variable, remove inconsistent values from neighbours' domains immediately."),
]

print("=" * 60)
print("Q4 — 20 Important Questions & Answers")
print("=" * 60)
for i, (q, a) in enumerate(qa_list, 1):
    print(f"\nQ{i:>2}. {q}")
    print(f"  A: {a}")

print("\n" + "=" * 60)
print("End of notebook.")
print("=" * 60)
