# Experimental Design: Regret-Optimal Exploration in Repeated Alternating-Offers Bargaining

**Version**: 1.0  
**Date**: 2026-02-03  
**Target**: AAMAS 2026 Empirical Validation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Experimental Setup](#2-experimental-setup)
3. [Main Experiments](#3-main-experiments)
4. [Ablation Studies](#4-ablation-studies)
5. [Robustness Analysis](#5-robustness-analysis)
6. [Computational Performance](#6-computational-performance)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Visualization Protocols](#8-visualization-protocols)
9. [Statistical Analysis](#9-statistical-analysis)
10. [Reproducibility Checklist](#10-reproducibility-checklist)

---

## 1. Overview

###1.1 Research Questions

**RQ1**: Does TSB achieve lower regret than baseline algorithms?  
**RQ2**: Does exploiting bargaining structure ($B$) improve performance?  
**RQ3**: How does TSB perform across different opponent distributions?  
**RQ4**: Is the theoretical regret bound $O(\sqrt{KT/B})$ validated empirically?  
**RQ5**: What is the computational overhead of TSB vs. simpler methods?

### 1.2 Hypotheses

**H1**: TSB achieves 30-50% lower cumulative regret than UCB1 after 1000 episodes.  
**H2**: Regret grows as $O(\sqrt{T})$ empirically, consistent with theory.  
**H3**: Performance gain increases with $B$ (strong bargaining structure).  
**H4**: TSB is robust to opponent distribution misspecification.  
**H5**: Runtime is within 2× of UCB1 (acceptable overhead).

---

## 2. Experimental Setup

### 2.1 Negotiation Environment

**Parameters**:
- Number of issues: $d = 3$
- Offer space: $X = [0,1]^3$ (simplex constraint: $\sum x_i = 1$)
- Deadline: $T_{\max} = 20$ rounds
- Discount factor: $\delta = 0.95$
- Disagreement values: $d_1 = d_2 = 0.1$

**Utility Functions**:
- Agent (learner): $u_1(x) = w_1^T x$ where $w_1 = [0.5, 0.3, 0.2]$ (fixed)
- Opponent: $u_2(x; \theta) = w_2(\theta)^T x$ where $w_2(\theta)$ depends on type

### 2.2 Opponent Types

| Type $\theta$ | Utility Weights $w_2(\theta)$ | Strategy | Concession Pattern |
|---------------|-------------------------------|----------|-------------------|
| **Conceder** | $[0.2, 0.5, 0.3]$ | Linear concession | $\alpha = 1.5$ |
| **Hardliner** | $[0.3, 0.2, 0.5]$ | Holds firm until late | $\beta = 5$ |
| **Tit-for-Tat** | $[0.4, 0.3, 0.3]$ | Mirrors agent | $\gamma = 0.9$ |
| **Boulware** | $[0.25, 0.35, 0.4]$ | Slow then sharp concession | Cubic |

**Opponent Type Distribution $\pi$**:
- **Uniform**: $\pi = [0.25, 0.25, 0.25, 0.25]$
- **Skewed**: $\pi = [0.5, 0.3, 0.15, 0.05]$ (Conceder-heavy)
- **Bimodal**: $\pi = [0.4, 0.1, 0.1, 0.4]$ (Conceder + Boulware)

### 2.3 Baseline Algorithms

**B1: Thompson Sampling for Bargaining (TSB)** [Our Method]
- Dirichlet prior: $\text{Dir}(1, 1, 1, 1)$ (uniform)
- Belief updates: Trajectory-based likelihood (Section 4.2 of TECHNICAL_SPEC.md)

**B2: UCB1 for Bargaining**
- Standard UCB1 with exploration bonus: $\sqrt{2 \log T / N_k(t)}$
- Type selection: $\arg\max_k \left[\hat{\mu}_k + c\sqrt{\frac{2\log t}{N_k(t)}}\right]$

**B3: ε-Greedy for Bargaining**
- Exploration rate: $\epsilon(t) = \min(1, 5K / t)$ (decaying)
- Type selection: Random with probability $\epsilon(t)$, greedy otherwise

**B4: Fixed Strategy**
- Always assumes opponent is most common type (Conceder in uniform distribution)
- No learning

**B5: Random**
- Selects opponent type uniformly at random each episode
- Lower bound baseline

### 2.4 Implementation Details

**Programming Language**: Python 3.8+

**Key Libraries**:
```python
numpy==1.24.0      # Numerical computations
scipy==1.10.0      # Statistical functions
matplotlib==3.7.0  # Plotting
seaborn==0.12.0    # Statistical visualization
tqdm==4.65.0       # Progress bars
```

**Random Seeds**: 10 seeds per condition (0-9) for statistical robustness

**Hardware**: Google Colab (CPU) or equivalent
- Expected runtime: 15-30 minutes for full experiment battery

---

## 3. Main Experiments

### 3.1 Experiment 1: Regret Comparison

**Goal**: Validate **H1** (TSB achieves lower regret)

**Setup**:
- Horizon: $T = 1000$ episodes
- Opponent distribution: Uniform
- Metrics: Cumulative regret, regret per episode
- Seeds: 10

**Procedure**:
```python
for algorithm in [TSB, UCB1, EpsilonGreedy, FixedStrategy, Random]:
    for seed in range(10):
        set_seed(seed)
        cumulative_regret = []
        
        for t in range(1000):
            # Sample opponent type
            theta_true = sample_opponent(pi_uniform)
            
            # Algorithm selects believed type
            theta_believed = algorithm.select_type()
            
            # Play negotiation
            utility = play_negotiation(theta_believed, theta_true)
            
            # Compute regret
            optimal_utility = oracle_utility(theta_true)
            regret = optimal_utility - utility
            cumulative_regret.append(regret)
            
            # Update algorithm
            algorithm.update(outcome)
        
        save_results(algorithm, seed, cumulative_regret)
```

**Expected Output**:
- **Figure 1**: Cumulative regret over time (5 curves, shaded std error)
- **Table 1**: Final regret at $T=1000$ with 95% confidence intervals

| Algorithm | Cumulative Regret | Relative to TSB |
|-----------|-------------------|-----------------|
| TSB | 127 ± 15 | 1.0× |
| UCB1 | 312 ± 28 | 2.46× |
| ε-Greedy | 418 ± 35 | 3.29× |
| Fixed Strategy | 589 ± 41 | 4.64× |
| Random | 742 ± 52 | 5.84× |

---

### 3.2 Experiment 2: Scaling with Horizon

**Goal**: Validate **H2** (Regret grows as $O(\sqrt{T})$)

**Setup**:
- Horizons: $T \in \{100, 200, 500, 1000, 2000, 5000\}$
- Opponent distribution: Uniform
- Algorithms: TSB, UCB1
- Seeds: 10 per horizon

**Analysis**:
Fit power law: $R_T = c \cdot T^\alpha$

Expected: $\alpha \approx 0.5$ for TSB (theory predicts $\sqrt{T}$)

**Expected Output**:
- **Figure 2**: Log-log plot of regret vs. horizon
- **Table 2**: Fitted exponents

| Algorithm | Fitted $\alpha$ | 95% CI | Theory |
|-----------|-----------------|--------|--------|
| TSB | 0.51 | [0.48, 0.54] | 0.5 |
| UCB1 | 0.53 | [0.49, 0.57] | 0.5 |

---

### 3.3 Experiment 3: Bargaining Structure Exploitation

**Goal**: Validate **H3** (Performance improves with $B$)

**Setup**:
- Vary opponent variance $\text{Var}_t[u_2]$ to manipulate $B$
- Three conditions:
  - **Low $B$** ($B \approx 10$): Flat opponent utilities
  - **Medium $B$** ($B \approx 100$): Default opponents
  - **High $B$** ($B \approx 1000$): Highly time-sensitive opponents
- Horizon: $T = 1000$
- Seeds: 10

**Expected Output**:
- **Figure 3**: Regret vs. $B$ (TSB shows stronger improvement than UCB1)

| $B$ | TSB Regret | UCB1 Regret | TSB Advantage |
|-----|------------|-------------|---------------|
| 10 | 305 ± 32 | 341 ± 29 | 11% |
| 100 | 127 ± 15 | 312 ± 28 | 59% |
| 1000 | 45 ± 8 | 287 ± 31 | 84% |

**Interpretation**: As $B$ increases (strong structure), TSB gap widens.

---

### 3.4 Experiment 4: Opponent Distribution Robustness

**Goal**: Validate **H4** (Robustness to distribution shift)

**Setup**:
- Train on Uniform, test on Skewed and Bimodal
- Horizon: $T = 1000$
- Seeds: 10

**Procedure**:
```python
# Train phase
train_dist = uniform
for t in range(500):
    theta = sample_opponent(train_dist)
    algorithm.update(play_negotiation(theta))

# Test phase (distribution shift)
test_dist = skewed  # or bimodal
test_regret = []
for t in range(500):
    theta = sample_opponent(test_dist)
    regret = compute_regret(algorithm, theta)
    test_regret.append(regret)
```

**Expected Output**:
- **Table 3**: Regret under distribution shift

| Algorithm | Train (Uniform) | Test (Skewed) | Test (Bimodal) |
|-----------|-----------------|---------------|----------------|
| TSB | 127 ± 15 | 145 ± 18 | 139 ± 17 |
| UCB1 | 312 ± 28 | 367 ± 35 | 352 ± 31 |

**Interpretation**: TSB maintains advantage under distribution shift (graceful degradation).

---

## 4. Ablation Studies

### 4.1 Ablation 1: Prior Sensitivity

**Goal**: Test sensitivity to prior choice

**Setup**:
- Priors: $\text{Dir}(0.1, 0.1, 0.1, 0.1)$ (diffuse), $\text{Dir}(1, 1, 1, 1)$ (uniform), $\text{Dir}(5, 5, 5, 5)$ (concentrated)
- Horizon: $T = 1000$
- Seeds: 10

**Expected Output**:
- **Figure 4**: Regret curves for different priors

| Prior | Cumulative Regret |
|-------|-------------------|
| Diffuse | 138 ± 18 |
| Uniform | 127 ± 15 |
| Concentrated | 119 ± 14 |

**Interpretation**: Reasonable robustness; concentrated prior helps if prior knowledge available.

---

### 4.2 Ablation 2: Belief Update Method

**Goal**: Compare trajectory-based vs. outcome-based likelihood

**Setup**:
- **Trajectory-based**: $L(\theta \mid \{x_1, \ldots, x_t\})$ (default)
- **Outcome-based**: $L(\theta \mid x_{\text{final}}, t_{\text{final}})$ (simpler)
- Horizon: $T = 1000$
- Seeds: 10

**Expected Output**:
- **Table 4**: Regret comparison

| Update Method | Cumulative Regret | Runtime |
|---------------|-------------------|---------|
| Trajectory-based | 127 ± 15 | 18 min |
| Outcome-based | 156 ± 19 | 12 min |

**Interpretation**: Trajectory-based uses more information → lower regret, but slower.

---

### 4.3 Ablation 3: Myopic vs. Optimal Best Response

**Goal**: Validate myopic approximation (Assumption 2.1)

**Setup**:
- **Myopic**: $\text{MBR}(t, \theta)$ (default)
- **Optimal**: Solve full game tree via backward induction
- Horizon: $T = 100$ (optimal is expensive)
- Seeds: 5

**Expected Output**:
- **Table 5**: Utility comparison

| Method | Avg Utility | Computation Time |
|--------|-------------|------------------|
| Myopic | 0.847 ± 0.03 | 0.02s per negotiation |
| Optimal | 0.863 ± 0.02 | 2.5s per negotiation |

**Interpretation**: Myopic is 98% of optimal utility, 125× faster → good trade-off.

---

### 4.4 Ablation 4: Impact of Deadline $T_{\max}$

**Goal**: Test performance under varying time pressure

**Setup**:
- Deadlines: $T_{\max} \in \{5, 10, 20, 50\}$
- Horizon: $T = 1000$
- Seeds: 10

**Expected Output**:
- **Figure 5**: Regret vs. $T_{\max}$

| $T_{\max}$ | TSB Regret | UCB1 Regret |
|------------|------------|-------------|
| 5 | 89 ± 12 | 267 ± 25 |
| 10 | 105 ± 14 | 289 ± 27 |
| 20 | 127 ± 15 | 312 ± 28 |
| 50 | 158 ± 19 | 341 ± 31 |

**Interpretation**: TSB advantage strongest under high time pressure (low $T_{\max}$).

---

## 5. Robustness Analysis

### 5.1 Experiment 5: Noisy Observations

**Goal**: Test robustness to stochastic opponent responses

**Setup**:
- Add noise to opponent acceptance decisions: $\Pr[\text{accept}] = \sigma(u_2(x) - \bar{u}_2 + \epsilon)$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$
- Noise levels: $\sigma \in \{0, 0.05, 0.1, 0.2\}$
- Horizon: $T = 1000$
- Seeds: 10

**Expected Output**:
- **Figure 6**: Regret vs. noise level

| Noise $\sigma$ | TSB Regret | UCB1 Regret |
|----------------|------------|-------------|
| 0.0 | 127 ± 15 | 312 ± 28 |
| 0.05 | 139 ± 17 | 328 ± 30 |
| 0.1 | 165 ± 21 | 354 ± 33 |
| 0.2 | 218 ± 28 | 401 ± 39 |

**Interpretation**: Both degrade gracefully; TSB maintains advantage.

---

### 5.2 Experiment 6: Unknown Number of Types

**Goal**: Test misspecified $K$ (model assumes $K=4$, true $K$ varies)

**Setup**:
- True number of types: $K_{\text{true}} \in \{2, 3, 4, 5, 6\}$
- Model assumes: $K_{\text{model}} = 4$
- Horizon: $T = 1000$
- Seeds: 10

**Expected Output**:
- **Table 6**: Regret under misspecified $K$

| $K_{\text{true}}$ | TSB Regret | UCB1 Regret |
|-------------------|------------|-------------|
| 2 | 98 ± 11 | 245 ± 24 |
| 3 | 115 ± 13 | 289 ± 27 |
| 4 | 127 ± 15 | 312 ± 28 |
| 5 | 152 ± 18 | 346 ± 32 |
| 6 | 189 ± 23 | 389 ± 37 |

**Interpretation**: Performance degrades smoothly with model misspecification.

---

### 5.3 Experiment 7: Non-Stationary Opponents

**Goal**: Test performance when opponent distribution drifts

**Setup**:
- Opponent distribution shifts linearly: $\pi(t) = (1-\alpha(t)) \pi_0 + \alpha(t) \pi_1$ where $\alpha(t) = t/T$
- Start: Uniform ($\pi_0$), End: Skewed ($\pi_1$)
- Horizon: $T = 1000$
- Use TSB with sliding window update (window size $W = 100$)
- Seeds: 10

**Expected Output**:
- **Figure 7**: Regret over time (stationary vs. non-stationary)

| Algorithm | Stationary Regret | Non-Stationary Regret |
|-----------|-------------------|------------------------|
| TSB (standard) | 127 ± 15 | 198 ± 24 |
| TSB (sliding window) | 127 ± 15 | 164 ± 19 |
| UCB1 | 312 ± 28 | 389 ± 35 |

**Interpretation**: Sliding window TSB adapts better to drift.

---

## 6. Computational Performance

### 6.1 Experiment 8: Runtime Analysis

**Goal**: Validate **H5** (Runtime overhead acceptable)

**Setup**:
- Measure wall-clock time for 1000 episodes
- Hardware: Google Colab CPU (Intel Xeon, 2 cores)
- Seeds: 5

**Expected Output**:
- **Table 7**: Runtime comparison

| Algorithm | Total Time (min) | Time per Episode (ms) | Relative to TSB |
|-----------|------------------|-----------------------|-----------------|
| TSB | 18.2 ± 1.3 | 1092 | 1.0× |
| UCB1 | 12.5 ± 0.9 | 750 | 0.69× |
| ε-Greedy | 11.8 ± 0.8 | 708 | 0.65× |

**Interpretation**: TSB is 1.5× slower but achieves 2.5× lower regret → worthwhile trade-off.

---

### 6.2 Experiment 9: Scalability

**Goal**: Test scalability to larger problems

**Setup**:
- Vary problem size:
  - Issues: $d \in \{2, 3, 5, 10\}$
  - Opponent types: $K \in \{2, 4, 8, 16\}$
- Horizon: $T = 500$ (shorter for larger problems)
- Seeds: 5

**Expected Output**:
- **Figure 8**: Runtime vs. $(d, K)$

| $(d, K)$ | TSB Time (min) | UCB1 Time (min) |
|----------|----------------|-----------------|
| (2, 2) | 5.1 | 3.8 |
| (3, 4) | 9.2 | 6.4 |
| (5, 8) | 23.7 | 15.2 |
| (10, 16) | 68.3 | 41.7 |

**Interpretation**: Both scale polynomially; TSB overhead remains ~1.5×.

---

## 7. Evaluation Metrics

### 7.1 Primary Metrics

**M1: Cumulative Regret**
$$R_T = \sum_{t=1}^T [V^*(\theta_t, 1) - u_1(x_t)]$$

**M2: Average Per-Episode Regret**
$$\bar{r} = \frac{1}{T} R_T$$

**M3: Agreement Rate**
$$\text{AR} = \frac{1}{T} \sum_{t=1}^T \mathbb{1}[\text{agreement reached in episode } t]$$

**M4: Average Rounds to Agreement**
$$\bar{t}_{\text{agree}} = \frac{1}{\sum_t \mathbb{1}[\text{agree}_t]} \sum_{t: \text{agree}_t} t_t$$

### 7.2 Secondary Metrics

**M5: Type Identification Accuracy**
$$\text{Acc}_t = \mathbb{1}[\hat{\theta}_t = \theta_t^{\text{true}}]$$

**M6: Posterior Concentration**
$$\text{Entropy}(\pi_t) = -\sum_{k=1}^K \pi_t(k) \log \pi_t(k)$$

**M7: Bayesian Regret**
$$\text{BR}_T = \mathbb{E}_{\theta \sim \pi}[R_T(\theta)]$$

---

## 8. Visualization Protocols

### 8.1 Regret Curves

**Figure Specification**:
```python
fig, ax = plt.subplots(figsize=(8, 6))

for algorithm in algorithms:
    mean_regret = np.mean(results[algorithm], axis=0)  # Average over seeds
    std_regret = np.std(results[algorithm], axis=0)
    
    ax.plot(mean_regret, label=algorithm, linewidth=2)
    ax.fill_between(
        range(T), 
        mean_regret - std_regret, 
        mean_regret + std_regret,
        alpha=0.2
    )

ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Cumulative Regret', fontsize=14)
ax.set_title('Regret Comparison', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

### 8.2 Bar Charts with Error Bars

**Figure Specification**:
```python
fig, ax = plt.subplots(figsize=(10, 6))

algorithms = ['TSB', 'UCB1', 'ε-Greedy', 'Fixed', 'Random']
means = [final_regret[alg].mean() for alg in algorithms]
stds = [final_regret[alg].std() for alg in algorithms]
cis = [1.96 * std / np.sqrt(n_seeds) for std in stds]  # 95% CI

x_pos = np.arange(len(algorithms))
ax.bar(x_pos, means, yerr=cis, capsize=5, alpha=0.7, color='steelblue')
ax.set_xticks(x_pos)
ax.set_xticklabels(algorithms, fontsize=12)
ax.set_ylabel('Final Cumulative Regret', fontsize=14)
ax.set_title('Algorithm Comparison (T=1000)', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
```

### 8.3 Heatmaps

**Figure Specification** (for $(d, K)$ scalability):
```python
import seaborn as sns

runtime_matrix = np.array([...])  # Shape: (len(d_values), len(K_values))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    runtime_matrix, 
    annot=True, 
    fmt='.1f', 
    cmap='YlOrRd',
    xticklabels=K_values,
    yticklabels=d_values,
    cbar_kws={'label': 'Runtime (min)'}
)
ax.set_xlabel('Number of Opponent Types (K)', fontsize=14)
ax.set_ylabel('Number of Issues (d)', fontsize=14)
ax.set_title('TSB Runtime Scalability', fontsize=16)
plt.tight_layout()
```

---

## 9. Statistical Analysis

### 9.1 Significance Testing

**Paired t-test** for comparing TSB vs. baselines:

```python
from scipy.stats import ttest_rel

# Null hypothesis: TSB and UCB1 have same regret
tsb_regret = final_regret['TSB']  # Shape: (n_seeds,)
ucb1_regret = final_regret['UCB1']

t_stat, p_value = ttest_rel(tsb_regret, ucb1_regret)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant: {'YES' if p_value < 0.05 else 'NO'}")
```

**Expected output**:
```
t-statistic: -8.742
p-value: 0.0001
Significant: YES
```

### 9.2 Effect Size

**Cohen's d**:
$$d = \frac{\bar{R}_{\text{TSB}} - \bar{R}_{\text{UCB1}}}{\sqrt{(\sigma_{\text{TSB}}^2 + \sigma_{\text{UCB1}}^2)/2}}$$

Expected: $d \approx 3.2$ (very large effect size)

### 9.3 Confidence Intervals

Report all metrics with 95% confidence intervals:
$$\text{CI}_{95\%} = \bar{x} \pm 1.96 \frac{\sigma}{\sqrt{n}}$$

---

## 10. Reproducibility Checklist

### 10.1 Code Availability

- [ ] Full source code on GitHub
- [ ] Google Colab notebook (one-click run)
- [ ] Docker container (frozen environment)
- [ ] requirements.txt with pinned versions

### 10.2 Data Availability

- [ ] Random seeds documented
- [ ] Opponent type specifications
- [ ] Raw results (CSV files)
- [ ] Processed results (aggregated statistics)

### 10.3 Documentation

- [ ] README with quick start
- [ ] TECHNICAL_SPEC.md with mathematical details
- [ ] EXPERIMENTS.md (this file) with protocols
- [ ] Inline code comments
- [ ] Docstrings for all functions

### 10.4 Validation

- [ ] Unit tests for algorithms
- [ ] Integration tests for environment
- [ ] Regression tests for key results
- [ ] Continuous integration (GitHub Actions)

---

## Appendix A: Experimental Timeline

**Total Runtime Estimate** (Google Colab CPU):

| Experiment | Runtime | Seeds | Total |
|------------|---------|-------|-------|
| Exp 1: Regret Comparison | 18 min | 10 | 3.0 hrs |
| Exp 2: Scaling | 15 min | 10 | 2.5 hrs |
| Exp 3: Structure | 18 min | 10 | 3.0 hrs |
| Exp 4: Distribution Robustness | 18 min | 10 | 3.0 hrs |
| Ablations 1-4 | 10 min | 10 | 1.7 hrs |
| Exp 5-7: Robustness | 20 min | 10 | 3.3 hrs |
| Exp 8-9: Performance | 5 min | 5 | 0.4 hrs |

**Total**: ~17 hours

**With parallelization** (5 seeds in parallel): ~6 hours

**Recommendation**: Run experiments in batches overnight on Colab.

---

## Appendix B: Result Archiving

All results saved to `results/` directory:

```
results/
├── raw/                         # Raw per-seed data
│   ├── exp1_regret_tsb_seed0.csv
│   ├── exp1_regret_ucb1_seed0.csv
│   └── ...
├── aggregated/                  # Aggregated statistics
│   ├── exp1_summary.csv
│   └── ...
├── plots/                       # Publication-ready figures
│   ├── fig1_regret_comparison.pdf
│   ├── fig2_scaling.pdf
│   └── ...
└── metadata.json                # Experiment config & timestamp
```

**Naming convention**: `exp{N}_{metric}_{algorithm}_seed{S}.csv`

---

**Version History**:
- v1.0 (2026-02-03): Initial experimental design for AAMAS 2026 submission

---

**Questions?** Open an issue on GitHub or email the authors.
