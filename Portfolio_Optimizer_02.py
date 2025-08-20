"""
Portfolio Optimizer using Evolutionary Algorithm
------------------------------------------------

This script implements a basic portfolio optimization strategy using an evolutionary algorithm.
It simulates natural selection to evolve portfolio allocations that aim to maximize return
while controlling for risk. Essentially, it tries to find the highest-return portfolio it can,
but penalizes portfolios that are too concentrated in any one asset.

Key Features:
- Randomly generates an initial population of portfolio allocations across four assets.
- Evaluates each portfolio using a simple fitness function: expected return minus a risk penalty.
- Selects the top-performing portfolios to act as parents for the next generation.
- Creates new portfolios (children) by combining and mutating the parent allocations.
- Evolves the population over multiple generations to converge on an optimal portfolio.

The output displays the best-performing portfolio found during the simulation, including:
- Percentage allocation to each asset.
- Estimated expected return.
- Risk metric (standard deviation of weights, for demonstration purposes as a stand-in for risk).
- In the real world, portfolio risk is usually measure by the standard deviation of returns,
- or more precisely by portfolio variance, which takes into account asset correlations via a covariance matrix.
- Since this is just a demo and we don't have historical price data or covariance between assets,
- we want a simple 'penalty' to acoid extreme weight distributions (like putting all the money in one asset).
- If the weights are evenly spread (like all 25%), the standard deviation is low.
- If all weight is in one asset and the rest are 0%, standard deviation is high.

This project is a simplified example meant for educational or prototyping use and can be extended
to handle real-world financial data, additional constraints, or more advanced risk models.

Bonus Technical Details
    Portfolio Return = w₁*r₁ + w₂*r₂ + ... + wₙ*rₙ
    w₁, w₂, ..., wₙ = weights (allocations) to each asset
    r₁, r₂, ..., rₙ = expected returns of each asset
"""

import numpy as np

# Let's say we have 4 assets
num_assets = 4
population_size = 20
generations = 50

# Create an initial population of random portfolios
def create_population(size, num_assets):
    return np.random.dirichlet(np.ones(num_assets), size=size)

# A simple fitness function: let's just say higher returns are better, and we penalize risk a bit
def fitness(portfolio):
    returns = np.dot(portfolio, [0.1, 0.12, 0.14, 0.08])  # pretend returns
    risk = np.std(portfolio)  # just a simple risk measure
    return returns - 0.2 * risk

# Evolutionary steps: selection, crossover, mutation
def evolve(population, fitness_fn):
    asset_names = ["Asset A", "Asset B", "Asset C", "Asset D"]
    expected_returns = [0.1, 0.12, 0.14, 0.08]

    # Print header
    print("\n📊 Portfolio Evolution by Generation")
    print(f"{'Gen':>4} | " + " | ".join(f"{name:^8}" for name in asset_names) + " |  Return  |  Risk")
    print("-" * 68)

    for gen in range(1, generations + 1):
        scores = np.array([fitness_fn(ind) for ind in population])
        top_indices = scores.argsort()[-population_size // 2:]
        parents = population[top_indices]

        # Get the best individual in this generation
        best_index = top_indices[-1]
        best_individual = population[best_index]
        portfolio_return = np.dot(best_individual, expected_returns)
        portfolio_risk = np.std(best_individual)

        # Print row
        print(f"{gen:>4} | " + " | ".join(f"{weight:>8.2%}" for weight in best_individual) +
              f" | {portfolio_return:>8.2%} | {portfolio_risk:>6.4f}")

        # Create next generation
        children = []
        while len(children) < population_size:
            indices = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[indices[0]], parents[indices[1]]
            child = (parent1 + parent2) / 2
            child += np.random.normal(0, 0.05, size=num_assets)
            child = np.abs(child) / np.sum(child)  # normalize
            children.append(child)
        population = np.array(children)

    return population

# Run the evolution
population = create_population(population_size, num_assets)
best_portfolios = evolve(population, fitness)

# Show the best portfolio
best_portfolio = max(best_portfolios, key=fitness)
# Define asset names and (optional) expected returns
asset_names = ["Asset A", "Asset B", "Asset C", "Asset D"]
expected_returns = [0.1, 0.12, 0.14, 0.08]  # Same as used in fitness()

# Print allocations
print("\n📊 Best Portfolio Allocation:")
for name, weight in zip(asset_names, best_portfolio):
    print(f"  {name}: {weight:.2%}")

# Optionally print metrics
portfolio_return = np.dot(best_portfolio, expected_returns)
portfolio_risk = np.std(best_portfolio)

print(f"\n📈 Expected Return: {portfolio_return:.2%}")
print(f"⚠️  Risk (Std Dev of Weights): {portfolio_risk:.4f}")