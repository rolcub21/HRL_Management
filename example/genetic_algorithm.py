import os
import json
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from small_rooms_env import SmallRoomsEnv

RUN_ID = f"ga_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
OUT_DIR = os.path.join("results", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)


def random_assignment(env):
    """Create a valid chromosome: one unique storage cell per block."""
    return random.sample(env.storage_positions, k=env.number_blocks)


def evaluate_assignment(env, chromosome):
    """
    Evaluate one chromosome.
    Assumes env.simulate_assignment accepts a list of storage positions.
    """
    return env.simulate_assignment(chromosome)


def crossover(parent1, parent2):
    """One-point crossover."""
    size = len(parent1)
    if size < 3:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, size - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def repair_assignment(chromosome, env):
    """
    Repair duplicates so each storage position is used at most once.
    """
    used = set()
    repaired = []
    missing = [p for p in env.storage_positions if p not in chromosome]
    miss_idx = 0

    for gene in chromosome:
        if gene not in used:
            repaired.append(gene)
            used.add(gene)
        else:
            repaired.append(missing[miss_idx])
            used.add(missing[miss_idx])
            miss_idx += 1

    return repaired


def mutate_swap(chromosome, mutation_rate=0.1):
    """
    Swap-mutation preserves uniqueness of assignments.
    """
    child = chromosome.copy()
    if random.random() < mutation_rate and len(child) >= 2:
        i, j = random.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
    return child


def ga_run(
    num_runs=3,
    pop_size=10,
    generations=20,
    elite_count=2,
    mutation_rate=0.1,
):
    import json
    import csv

    all_best_fitness = []
    all_avg_fitness = []
    per_run_best = []
    per_run_final_best = []
    per_run_final_avg = []

    best_sol_overall = None
    best_fitness_overall = float("-inf")

    for run in range(num_runs):
        seed = run
        random.seed(seed)
        np.random.seed(seed)

        env = SmallRoomsEnv()

        sample_chromosome = random_assignment(env)
        chromosome_length = len(sample_chromosome)

        print(f"\n=== Starting GA Run {run + 1}/{num_runs} ===")
        print(f"[INFO] Seed = {seed}")
        print(f"[INFO] Chromosome length = {chromosome_length} (= number_of_blocks)")

        population = [random_assignment(env) for _ in range(pop_size)]

        best_fit_run = float("-inf")
        best_sol_run = None

        best_fitnesses = []
        avg_fitnesses = []

        for g in range(generations):
            fitness_scores = [evaluate_assignment(env, individual) for individual in population]

            current_best = max(fitness_scores)
            current_avg = float(np.mean(fitness_scores))

            best_fitnesses.append(current_best)
            avg_fitnesses.append(current_avg)

            best_idx = int(np.argmax(fitness_scores))
            if current_best > best_fit_run:
                best_fit_run = current_best
                best_sol_run = population[best_idx].copy()

            sorted_pop = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)

            elites = [ind.copy() for ind, _ in sorted_pop[:elite_count]]
            survivors = [ind for ind, _ in sorted_pop[: pop_size // 2]]

            new_population = elites[:]
            while len(new_population) < pop_size:
                parent1, parent2 = random.sample(survivors, 2)

                child1, child2 = crossover(parent1, parent2)

                child1 = repair_assignment(child1, env)
                child1 = mutate_swap(child1, mutation_rate=mutation_rate)
                child1 = repair_assignment(child1, env)

                child2 = repair_assignment(child2, env)
                child2 = mutate_swap(child2, mutation_rate=mutation_rate)
                child2 = repair_assignment(child2, env)

                new_population.extend([child1, child2])

            population = new_population[:pop_size]

            if (g + 1) % 1 == 0 or g == 0:
                print(
                    f"Run {run + 1} | Gen {g + 1:4d} | "
                    f"Best {current_best:8.2f} | Avg {current_avg:8.2f}"
                )

        print(f"Run {run + 1} completed. Best fitness in this run: {best_fit_run:.2f}")

        all_best_fitness.append(best_fitnesses)
        all_avg_fitness.append(avg_fitnesses)

        per_run_best.append(float(best_fit_run))
        per_run_final_best.append(float(best_fitnesses[-1]))
        per_run_final_avg.append(float(avg_fitnesses[-1]))

        if best_fit_run > best_fitness_overall:
            best_fitness_overall = best_fit_run
            best_sol_overall = best_sol_run

    # Aggregate across runs
    all_best_fitness = np.asarray(all_best_fitness, dtype=float)
    all_avg_fitness = np.asarray(all_avg_fitness, dtype=float)

    mean_best_fitness = np.mean(all_best_fitness, axis=0)
    std_best_fitness = np.std(all_best_fitness, axis=0)

    mean_avg_fitness = np.mean(all_avg_fitness, axis=0)
    std_avg_fitness = np.std(all_avg_fitness, axis=0)

    # Final-generation summary numbers for tables
    final_best_fitness_mean = float(mean_best_fitness[-1])
    final_best_fitness_std = float(std_best_fitness[-1])
    final_avg_fitness_mean = float(mean_avg_fitness[-1])
    final_avg_fitness_std = float(std_avg_fitness[-1])

    overall_best_mean = float(np.mean(per_run_best))
    overall_best_std = float(np.std(per_run_best))

    # Plot GA performance over generations
    plt.figure(figsize=(12, 6))

    plt.plot(mean_best_fitness, label="Mean Best Fitness", color="blue")
    plt.fill_between(
        range(generations),
        mean_best_fitness - std_best_fitness,
        mean_best_fitness + std_best_fitness,
        color="blue",
        alpha=0.2,
        label="Best Fitness Std Dev",
    )

    plt.plot(mean_avg_fitness, label="Mean Average Fitness", color="orange")
    plt.fill_between(
        range(generations),
        mean_avg_fitness - std_avg_fitness,
        mean_avg_fitness + std_avg_fitness,
        color="orange",
        alpha=0.2,
        label="Average Fitness Std Dev",
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness (Total Reward)")
    plt.title("GA Performance Over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("ga_performance_over_generations_seeds.png")
    plot_path = os.path.join(OUT_DIR, "ga_performance_over_generations_seeds.png")
    plt.savefig(plot_path)
    plt.close()

    # Save best overall solution
    print("\n===========================================================")
    print(f"Best fitness across all runs: {best_fitness_overall:.2f}")
    print("Best assignment found:")
    print(best_sol_overall)

    best_assignment_path = os.path.join(OUT_DIR, "best_assignment.pkl")
    with open(best_assignment_path, "wb") as f:
        pickle.dump(best_sol_overall, f)
    print("Best assignment saved to best_assignment.pkl")

    # -------- Added summary export block --------
    summary = {
        "num_runs": int(num_runs),
        "pop_size": int(pop_size),
        "generations": int(generations),
        "elite_count": int(elite_count),
        "mutation_rate": float(mutation_rate),
        "best_fitness_overall": float(best_fitness_overall),
        "overall_best_mean": overall_best_mean,
        "overall_best_std": overall_best_std,
        "final_best_fitness_mean": final_best_fitness_mean,
        "final_best_fitness_std": final_best_fitness_std,
        "final_avg_fitness_mean": final_avg_fitness_mean,
        "final_avg_fitness_std": final_avg_fitness_std,
        "per_run_best": per_run_best,
        "per_run_final_best": per_run_final_best,
        "per_run_final_avg": per_run_final_avg,
    }


    summary_json_path = os.path.join(OUT_DIR, "ga_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    runs_csv_path = os.path.join(OUT_DIR, "ga_runs.csv")
    with open(runs_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "best_fitness", "final_best_fitness", "final_avg_fitness"])
        for i in range(num_runs):
            writer.writerow([
                i,
                per_run_best[i],
                per_run_final_best[i],
                per_run_final_avg[i],
            ])

    summary_csv_path = os.path.join(OUT_DIR, "ga_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_runs", num_runs])
        writer.writerow(["pop_size", pop_size])
        writer.writerow(["generations", generations])
        writer.writerow(["elite_count", elite_count])
        writer.writerow(["mutation_rate", mutation_rate])
        writer.writerow(["best_fitness_overall", best_fitness_overall])
        writer.writerow(["overall_best_mean", overall_best_mean])
        writer.writerow(["overall_best_std", overall_best_std])
        writer.writerow(["final_best_fitness_mean", final_best_fitness_mean])
        writer.writerow(["final_best_fitness_std", final_best_fitness_std])
        writer.writerow(["final_avg_fitness_mean", final_avg_fitness_mean])
        writer.writerow(["final_avg_fitness_std", final_avg_fitness_std])

    print(f"Best assignment saved to {best_assignment_path}")
    print(f"Saved {plot_path}")
    print(f"Saved {summary_json_path}")
    print(f"Saved {summary_csv_path}")
    print(f"Saved {runs_csv_path}")
    # -------- End added block --------

    return {
        "best_assignment": best_sol_overall,
        "best_fitness": best_fitness_overall,
        "mean_best_fitness": mean_best_fitness,
        "std_best_fitness": std_best_fitness,
        "mean_avg_fitness": mean_avg_fitness,
        "std_avg_fitness": std_avg_fitness,
        "summary": summary,
    }

if __name__ == "__main__":
    ga_run(num_runs=3)