import json
import sys
import numpy as np

def analyze(json_path):
    with open(json_path) as f:
        data = json.load(f)

    base_pts = data["points"]["baseline"]
    exp_pts = data["points"]["experimental"]
    draws = data["points"]["draws"]
    total_games = base_pts + exp_pts + draws

    print(f"Total Games: {total_games}")
    print(f"Baseline: {base_pts} wins ({base_pts + 0.5 * draws} pts, {(base_pts + 0.5 * draws) / total_games * 100:.2f}%)")
    print(f"Experimental: {exp_pts} wins ({exp_pts + 0.5 * draws} pts, {(exp_pts + 0.5 * draws) / total_games * 100:.2f}%)")
    print(f"Draws: {draws} ({(draws / total_games) * 100:.2f}%)")

    # Pair results
    pair_results = data.get("pair_results", [])
    if not pair_results:
        # Reconstruct pairs from game_results
        pair_dict = {}
        for g in data.get("game_results", []):
            p_idx = g.get("pair_index")
            if p_idx is not None:
                pair_dict.setdefault(p_idx, []).append(g)
        deltas = []
        for p_idx, games in pair_dict.items():
            if len(games) == 2:
                # game 0 baseline score, game 1 baseline score
                # g["result_baseline_pov"]: 1.0 (base win), 0.0 (draw), -1.0 (exp win)
                # exp points in pair: (1 - res) / 2
                base_pair_score = sum((g["result_baseline_pov"] + 1.0) / 2.0 for g in games)
                exp_pair_score = sum((1.0 - g["result_baseline_pov"]) / 2.0 for g in games)
                deltas.append(exp_pair_score - base_pair_score)
    else:
        deltas = [p.get("experimental_points", 0) - p.get("baseline_points", 0) for p in pair_results]

    deltas = np.array(deltas)
    n_pairs = len(deltas)
    mean_delta = np.mean(deltas)
    std_err = np.std(deltas, ddof=1) / np.sqrt(n_pairs)

    # 10,000 bootstrap iterations
    np.random.seed(20260921)
    boot_means = []
    for _ in range(10000):
        sample = np.random.choice(deltas, size=n_pairs, replace=True)
        boot_means.append(np.mean(sample))

    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    exp_rate = (exp_pts + 0.5 * draws) / total_games
    if 0.0 < exp_rate < 1.0:
        elo_diff = -400.0 * np.log10(1.0 / exp_rate - 1.0)
    else:
        elo_diff = 0.0

    print("\n--- Paired Statistics ---")
    print(f"Opening pairs: {n_pairs}")
    print(f"Mean pair delta (exp - base): {mean_delta:+.4f} (per pair of 2 games)")
    print(f"Mean game delta: {mean_delta / 2.0:+.4f}")
    print(f"Normal 95% CI: [{mean_delta - 1.96 * std_err:.4f}, {mean_delta + 1.96 * std_err:.4f}]")
    print(f"Bootstrap 95% CI (10k resamples): [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Bootstrap 95% LCB (Lower Confidence Bound): {ci_low:.4f}")
    print(f"Estimated Elo diff: {elo_diff:+.1f}")
    print(f"Decision: {'PROMOTION ACCEPTED (LCB > 0)' if ci_low > 0 else 'NEUTRAL OR BELOW ZERO'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_paired_screen.py <json_path>")
        sys.exit(1)
    analyze(sys.argv[1])
