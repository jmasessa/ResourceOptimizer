"""
Real-World Portfolio Optimizer using Evolutionary Algorithm and Vanguard ETF Data
-------------------------------------------------------------------------------
This script uses historical daily prices of five Vanguard ETFs to optimize
portfolio allocations using an evolutionary algorithm. Now extended to run
through *every possible 5-ETF combination* from a master list to find the
best-performing combo based on return vs. risk (fitness score).

Key Features:
- Pulls daily adjusted close data for Vanguard ETFs from Yahoo Finance
- Calculates annualized expected returns and the covariance matrix of returns
- Uses an evolutionary algorithm to evolve portfolio allocations
- Fitness function: Expected return - 0.2 * Risk (std. dev. of portfolio returns)
- Compares all combinations of 5 ETFs from a larger list
- Displays top ETF sets based on best portfolio fitness
- Saves top results to CSV with allocation breakdowns

Formulas:
Portfolio Return:
    Return = wrights * mean daily returns
Risk:
    Risk = sqrt((weights^T) * covariance matrix * weights)
Fitness Score:
    Fitness = Return / Risk
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
import time
import csv
import os

# Configurable parameters
# --- 10 ETF list ---
master_etf_list = ["VOO", "VB", "VXUS", "BND", "VNQ", "VTI", "VEA", "VTV", "VUG", "VYM"]
"""# --- 20 ETF list ---
master_etf_list = [
     "VOO",   # S&P 500
     "VTI",   # Total US Stock Market
     "VB",    # Small-Cap
     "VTV",   # Value
     "VUG",   # Growth
     "VYM",   # High Dividend Yield
     "BND",   # Total Bond Market
     "BSV",   # Short-Term Bond
     "BLV",   # Long-Term Bond
     "VXUS",  # Total International Stock
     "VEA",   # Developed Markets ex-US
     "VWO",   # Emerging Markets
     "VNQ",   # Real Estate
     "VGT",   # Information Technology
     "VPU",   # Utilities
     "VFH",   # Financials
     "VIS",   # Industrials
     "VAW",   # Materials
     "VOX",   # Communication Services
     "VDC"    # Consumer Staples
 ]"""
""" --- 30 ETF list ---
master_etf_list = [
     "VOO", "VTI", "VB", "VTV", "VUG", "VYM", "BND", "BSV", "BLV", "VXUS",
     "VEA", "VWO", "VNQ", "VGT", "VPU", "VFH", "VIS", "VAW", "VOX", "VDC",
     "VHT",   # Health Care
     "VDE",   # Energy
     "VCR",   # Consumer Discretionary
     "MGK",   # Mega Cap Growth
     "MGV",   # Mega Cap Value
     "VIOO",  # Small-Cap 600
     "VIOG",  # Small-Cap Growth
     "VIOV",  # Small-Cap Value
     "VT",    # Global Total Market
     "ESGV"   # ESG US Stock
 ]"""

start_date = "2020-01-01"
end_date = "2025-01-01"
combo_size = 5
population_size = 30
generations = 50

# ----- FETCH DATA -----
def get_data(etfs):
    data = yf.download(etfs, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    prices = pd.concat([data[ticker]['Close'] for ticker in etfs], axis=1)
    prices.columns = etfs
    return prices.dropna()

# ----- FITNESS FUNCTION -----
def fitness(weights, mean_returns, cov_matrix):
    # Normalize weights before calculating return, risk, and fitness
    normalized_weights = weights / weights.sum()
    port_return = normalized_weights @ mean_returns
    port_risk = (normalized_weights.T @ cov_matrix @ normalized_weights) ** 0.5
    fitness_score = port_return / port_risk if port_risk != 0 else 0
    return port_return, port_risk, fitness_score

# ----- EVOLUTIONARY ALGORITHM -----
def evolve_portfolio(prices, tickers):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    best_result = []

    population = [np.random.dirichlet(np.ones(len(tickers))) for _ in range(population_size)]

    print("\n📊 Portfolio Evolution by Generation")
    print(" Gen | " + " | ".join([f"{ticker:^6}" for ticker in tickers]) + " | Return |  Risk  | Fitness")
    print("-" * (9 + 10 * len(tickers)))

    for gen in range(generations):
        scored_population = [(*fitness(ind, mean_returns, cov_matrix), ind) for ind in population]
        scored_population.sort(reverse=True, key=lambda x: x[2])
        top = scored_population[:population_size // 2]
        best = top[0]
        best_result.append(best)

        allocations = best[3] / sum(best[3])  # Normalize allocations before display
        row = f" {gen + 1:>3} | " + " | ".join([f"{w * 100:6.2f}" for w in allocations]) + f" | {best[0]*100:6.2f}% | {best[1]:6.4f} | {best[2]:6.4f}"
        print(row)

        children = []
        for _ in range(population_size):
            p1, p2 = random.sample(top, 2)
            child = (p1[3] + p2[3]) / 2
            mutation = np.random.normal(0, 0.05, len(tickers))
            child += mutation
            child = np.clip(child, 0, None)
            child /= child.sum()
            children.append(child)

        population = children

    return best_result[-1]  # Return final generation's best result

# ----- TEST COMBINATIONS -----
def run_all_combos(master_list, combo_size):
    all_combos = list(itertools.combinations(master_list, combo_size))
    results = []

    for combo in all_combos:
        try:
            prices = get_data(list(combo))
            best = evolve_portfolio(prices, list(combo))
            allocations = best[3] / sum(best[3])
            results.append({
                "Combo": combo,
                "Allocations": allocations,
                "Return": best[0],
                "Risk": best[1],
                "Fitness": best[2]
            })
        except Exception as e:
            print(f"❌ Error with combo {combo}: {e}")

    return sorted(results, key=lambda x: x['Fitness'], reverse=True)[:10]

# ----- MAIN -----
if __name__ == "__main__":
    print("🚀 Starting optimization across ETF combinations...")
    top_results = run_all_combos(master_etf_list, combo_size)

    print("\n✅ Finished testing. Top 10 combos with full allocation details:\n")
    output_data = []

    for i, result in enumerate(top_results, 1):
        print(f"🔹 Combo #{i}: {result['Combo']}")
        print("   Allocations:")
        for ticker, alloc in zip(result['Combo'], result['Allocations']):
            print(f"     - {ticker}: {alloc * 100:.2f}%")
        print(f"   📈 Return: {result['Return'] * 100:.2f}%")
        print(f"   ⚠️  Risk:   {result['Risk']:.4f}")
        print(f"   🧠 Fitness Score: {result['Fitness']:.4f}")
        print("-" * 60)

        # Prepare row for CSV
        row = {
            "Combo": ', '.join(result['Combo']),
            "Return": result['Return'],
            "Risk": result['Risk'],
            "Fitness": result['Fitness']
        }
        row.update({ticker: alloc for ticker, alloc in zip(result['Combo'], result['Allocations'])})
        output_data.append(row)

    # Save to CSV
    df_out = pd.DataFrame(output_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = f"top_etf_combos_{timestamp}.csv"
    df_out.to_csv(csv_name, index=False)
    print(f"\n💾 Results saved to {csv_name}")