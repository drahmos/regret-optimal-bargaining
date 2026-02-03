# Implementation Plan: Regret-Optimal Bargaining

**Version**: 1.0  
**Date**: 2026-02-03  
**Timeline**: 5-7 days to complete implementation

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Breakdown](#2-module-breakdown)
3. [Implementation Priority](#3-implementation-priority)
4. [Code Structure](#4-code-structure)
5. [Development Timeline](#5-development-timeline)
6. [Testing Strategy](#6-testing-strategy)
7. [Code Snippets](#7-code-snippets)
8. [Common Pitfalls](#8-common-pitfalls)

---

## 1. Architecture Overview

### 1.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT RUNNER                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ Config     │  │ Logging    │  │ Results    │           │
│  │ Manager    │  │ System     │  │ Storage    │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼─────────┐     ┌────────▼────────┐
│   ALGORITHMS      │     │  ENVIRONMENT    │
│                   │     │                 │
│ • TSB             │◄────┤ • Bargaining    │
│ • UCB1            │     │   Simulator     │
│ • ε-Greedy        │     │ • Opponent      │
│ • Fixed Strategy  │     │   Models        │
│ • Random          │     │                 │
└───────────────────┘     └─────────────────┘
          │                         │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │      UTILITIES          │
          │                         │
          │ • Metrics Calculator    │
          │ • Visualization         │
          │ • Statistical Tests     │
          └─────────────────────────┘
```

### 1.2 Key Design Principles

1. **Modularity**: Each component is independently testable
2. **Reproducibility**: Fixed random seeds, logged configuration
3. **Extensibility**: Easy to add new algorithms or opponent types
4. **Performance**: Vectorized operations where possible
5. **Clarity**: Prioritize readability over micro-optimizations

---

## 2. Module Breakdown

### 2.1 Environment Module (`src/environment/`)

**Purpose**: Simulate bilateral alternating-offers bargaining

**Files**:
- `bargaining_env.py` - Main environment class
- `opponent_models.py` - Opponent type implementations
- `utility_functions.py` - Utility computation

**Key Classes**:

```python
class BargainingEnvironment:
    """
    Simulates a single negotiation episode.
    """
    def __init__(self, n_issues=3, T_max=20, delta=0.95):
        pass
    
    def reset(self, opponent_type):
        """Start new negotiation with given opponent type."""
        pass
    
    def step(self, offer):
        """Agent makes offer, opponent responds."""
        pass
    
    def get_state(self):
        """Return current round, history, etc."""
        pass
    
    def is_done(self):
        """Check if negotiation ended."""
        pass

class OpponentModel:
    """
    Base class for opponent types.
    """
    def __init__(self, utility_weights, concession_params):
        pass
    
    def generate_offer(self, round, history):
        """Generate opponent's offer at given round."""
        pass
    
    def accept_offer(self, offer, round):
        """Decide whether to accept agent's offer."""
        pass
```

**Implementation Details**:
- Use NumPy for vectorized operations
- Store history as deque for efficiency
- Implement acceptance probability with logistic function

---

### 2.2 Algorithm Module (`src/algorithms/`)

**Purpose**: Implement TSB and baseline algorithms

**Files**:
- `base_algorithm.py` - Abstract base class
- `thompson_bargaining.py` - TSB implementation
- `ucb1.py` - UCB1 baseline
- `epsilon_greedy.py` - ε-Greedy baseline
- `fixed_strategy.py` - Fixed strategy baseline
- `random_baseline.py` - Random baseline

**Key Classes**:

```python
class BaseAlgorithm(ABC):
    """
    Abstract base for all algorithms.
    """
    @abstractmethod
    def select_type(self, episode):
        """Select believed opponent type."""
        pass
    
    @abstractmethod
    def update(self, outcome):
        """Update beliefs/statistics given outcome."""
        pass
    
    def reset(self):
        """Reset algorithm state."""
        pass

class ThompsonSamplingBargaining(BaseAlgorithm):
    """
    Thompson Sampling for Bargaining (TSB).
    """
    def __init__(self, n_types, prior_alpha=None):
        self.n_types = n_types
        self.alpha = prior_alpha or np.ones(n_types)
    
    def select_type(self, episode):
        """Sample from Dirichlet posterior."""
        pi_sample = np.random.dirichlet(self.alpha)
        type_idx = np.random.choice(self.n_types, p=pi_sample)
        return type_idx
    
    def update(self, outcome):
        """Bayesian update based on likelihood."""
        likelihoods = self._compute_likelihoods(outcome)
        self.alpha += likelihoods
    
    def _compute_likelihoods(self, outcome):
        """Compute L(theta_k | outcome) for each type."""
        pass
```

**Implementation Details**:
- TSB: Use SciPy for Dirichlet sampling
- UCB1: Implement with NumPy for efficiency
- Store per-type statistics (counts, means)

---

### 2.3 Theory Module (`src/theory/`)

**Purpose**: Compute theoretical quantities (regret bounds, oracle values)

**Files**:
- `regret_bounds.py` - Theoretical regret computation
- `oracle.py` - Optimal strategy computation
- `lower_bounds.py` - Worst-case analysis

**Key Functions**:

```python
def compute_oracle_utility(opponent_type, env_params):
    """
    Compute optimal utility against known opponent type.
    Uses backward induction.
    """
    pass

def compute_regret_bound(K, T, B, method='tsb'):
    """
    Compute theoretical regret bound.
    
    Args:
        K: Number of opponent types
        T: Number of episodes
        B: Bargaining structure parameter
        method: 'tsb', 'ucb1', 'generic'
    
    Returns:
        Upper bound on expected regret
    """
    if method == 'tsb':
        return np.sqrt(K * T * np.log(T) / B)
    elif method == 'generic':
        return np.sqrt(K * T * np.log(T))
    else:
        raise ValueError(f"Unknown method: {method}")

def estimate_B(opponent_models, env_params):
    """
    Estimate bargaining structure parameter B from opponent models.
    
    B = (d * K) / E[Var_t[u_2(x_t; theta)]]
    """
    pass
```

---

### 2.4 Utilities Module (`src/utils/`)

**Purpose**: Helper functions for metrics, visualization, statistics

**Files**:
- `metrics.py` - Regret, agreement rate, etc.
- `visualization.py` - Plotting functions
- `statistical_tests.py` - t-tests, effect sizes
- `logging_utils.py` - Experiment logging

**Key Functions**:

```python
def compute_cumulative_regret(utilities, oracle_utilities):
    """Compute cumulative regret over time."""
    return np.cumsum(oracle_utilities - utilities)

def plot_regret_curves(results_dict, save_path=None):
    """
    Plot regret curves for multiple algorithms.
    
    Args:
        results_dict: {algorithm_name: regret_array}
        save_path: Optional path to save figure
    """
    pass

def run_paired_ttest(method1_results, method2_results):
    """
    Run paired t-test between two methods.
    
    Returns:
        t_statistic, p_value, cohen_d
    """
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(method1_results, method2_results)
    cohen_d = compute_cohens_d(method1_results, method2_results)
    return t_stat, p_val, cohen_d
```

---

### 2.5 Experiments Module (`experiments/`)

**Purpose**: Run full experimental pipeline

**Files**:
- `run_experiment.py` - Single experiment runner
- `run_all_experiments.py` - Full battery
- `config.py` - Experiment configurations
- `analysis.py` - Post-processing results

**Key Functions**:

```python
def run_single_experiment(algorithm, opponent_dist, config, seed):
    """
    Run one experiment instance.
    
    Returns:
        Dictionary with results (regret, utilities, etc.)
    """
    np.random.seed(seed)
    
    # Initialize environment and algorithm
    env = BargainingEnvironment(**config['env_params'])
    alg = algorithm(**config['alg_params'])
    
    # Run T episodes
    results = {
        'regrets': [],
        'utilities': [],
        'types_selected': [],
        'types_true': []
    }
    
    for episode in range(config['T']):
        # Sample opponent type
        true_type = np.random.choice(
            len(opponent_dist), 
            p=opponent_dist
        )
        
        # Algorithm selects type
        believed_type = alg.select_type(episode)
        
        # Play negotiation
        utility = play_negotiation(
            env, 
            believed_type, 
            true_type
        )
        
        # Compute oracle utility
        oracle_utility = compute_oracle_utility(
            true_type, 
            config['env_params']
        )
        
        # Store results
        results['utilities'].append(utility)
        results['regrets'].append(oracle_utility - utility)
        results['types_selected'].append(believed_type)
        results['types_true'].append(true_type)
        
        # Update algorithm
        outcome = {
            'utility': utility,
            'type_true': true_type,
            'history': env.get_history()
        }
        alg.update(outcome)
    
    return results

def run_experiment_suite(config, n_seeds=10):
    """
    Run full suite of experiments across seeds.
    """
    all_results = {}
    
    for algorithm in config['algorithms']:
        alg_results = []
        for seed in range(n_seeds):
            result = run_single_experiment(
                algorithm, 
                config['opponent_dist'],
                config,
                seed
            )
            alg_results.append(result)
        all_results[algorithm.__name__] = alg_results
    
    return all_results
```

---

## 3. Implementation Priority

### Phase 1: Core Components (Days 1-2)
**Priority: CRITICAL**

1. ✅ **BargainingEnvironment** (`environment/bargaining_env.py`)
   - Single negotiation simulator
   - Test: Can run one negotiation end-to-end

2. ✅ **OpponentModels** (`environment/opponent_models.py`)
   - Implement 4 types: Conceder, Hardliner, Tit-for-Tat, Boulware
   - Test: Each type generates sensible offers

3. ✅ **BaseAlgorithm** (`algorithms/base_algorithm.py`)
   - Abstract interface
   - Test: Can instantiate subclasses

4. ✅ **ThompsonSamplingBargaining** (`algorithms/thompson_bargaining.py`)
   - Core TSB implementation
   - Test: Belief updates correctly

### Phase 2: Baselines (Day 3)
**Priority: HIGH**

5. ✅ **UCB1** (`algorithms/ucb1.py`)
6. ✅ **EpsilonGreedy** (`algorithms/epsilon_greedy.py`)
7. ✅ **FixedStrategy** (`algorithms/fixed_strategy.py`)
8. ✅ **RandomBaseline** (`algorithms/random_baseline.py`)

**Test**: All baselines run without errors

### Phase 3: Experiments (Days 4-5)
**Priority: HIGH**

9. ✅ **Experiment Runner** (`experiments/run_experiment.py`)
   - Single experiment execution
   - Test: Can run 100 episodes

10. ✅ **Configuration** (`experiments/config.py`)
    - Centralized config management
    - Test: Configs load correctly

11. ✅ **Main Experiments** (`experiments/run_all_experiments.py`)
    - Exp 1-4 from EXPERIMENTS.md
    - Test: Generates expected output files

### Phase 4: Analysis (Days 6-7)
**Priority: MEDIUM**

12. ✅ **Metrics** (`utils/metrics.py`)
13. ✅ **Visualization** (`utils/visualization.py`)
14. ✅ **Statistical Tests** (`utils/statistical_tests.py`)
15. ✅ **Analysis Pipeline** (`experiments/analysis.py`)

**Test**: Generates publication-ready plots

### Phase 5: Polish (Day 7)
**Priority: LOW**

16. ✅ Documentation strings
17. ✅ Code formatting (black, isort)
18. ✅ Type hints
19. ✅ README update with results

---

## 4. Code Structure

### 4.1 Directory Layout

```
regret-optimal-bargaining/
├── README.md
├── TECHNICAL_SPEC.md
├── EXPERIMENTS.md
├── IMPLEMENTATION.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── bargaining_env.py       # ~200 lines
│   │   ├── opponent_models.py      # ~300 lines
│   │   └── utility_functions.py    # ~50 lines
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base_algorithm.py       # ~80 lines
│   │   ├── thompson_bargaining.py  # ~150 lines
│   │   ├── ucb1.py                 # ~100 lines
│   │   ├── epsilon_greedy.py       # ~80 lines
│   │   ├── fixed_strategy.py       # ~50 lines
│   │   └── random_baseline.py      # ~40 lines
│   │
│   ├── theory/
│   │   ├── __init__.py
│   │   ├── regret_bounds.py        # ~100 lines
│   │   ├── oracle.py               # ~120 lines
│   │   └── lower_bounds.py         # ~80 lines
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py              # ~150 lines
│       ├── visualization.py        # ~250 lines
│       ├── statistical_tests.py    # ~100 lines
│       └── logging_utils.py        # ~80 lines
│
├── experiments/
│   ├── __init__.py
│   ├── run_experiment.py           # ~200 lines
│   ├── run_all_experiments.py      # ~300 lines
│   ├── config.py                   # ~150 lines
│   └── analysis.py                 # ~200 lines
│
├── tests/
│   ├── __init__.py
│   ├── test_environment.py         # ~200 lines
│   ├── test_algorithms.py          # ~250 lines
│   ├── test_theory.py              # ~100 lines
│   └── test_utils.py               # ~150 lines
│
├── notebooks/
│   ├── full_experiments.ipynb      # Main Colab notebook
│   ├── quick_test.ipynb            # 5-minute verification
│   └── ablation_studies.ipynb      # Ablations
│
└── results/                        # .gitignored
    ├── raw/
    ├── aggregated/
    └── plots/
```

**Total LOC Estimate**: ~3,000 lines (manageable for 1 week)

---

## 5. Development Timeline

### Day 1: Environment (6-8 hours)
- **Morning** (3-4 hrs):
  - [ ] Create directory structure
  - [ ] Implement `BargainingEnvironment`
  - [ ] Write unit tests for environment
  
- **Afternoon** (3-4 hrs):
  - [ ] Implement `OpponentModels` (all 4 types)
  - [ ] Test opponent behavior (visual inspection)
  - [ ] Debug edge cases

**Deliverable**: Working negotiation simulator

---

### Day 2: Core Algorithm (6-8 hours)
- **Morning** (3-4 hrs):
  - [ ] Implement `BaseAlgorithm`
  - [ ] Implement `ThompsonSamplingBargaining`
  - [ ] Write unit tests for TSB
  
- **Afternoon** (3-4 hrs):
  - [ ] Implement likelihood computation
  - [ ] Test belief updates (verify posteriors)
  - [ ] Optimize for performance

**Deliverable**: Working TSB implementation

---

### Day 3: Baselines (4-6 hours)
- **Morning** (2-3 hrs):
  - [ ] Implement UCB1
  - [ ] Implement ε-Greedy
  
- **Afternoon** (2-3 hrs):
  - [ ] Implement Fixed Strategy
  - [ ] Implement Random Baseline
  - [ ] Unit tests for all baselines

**Deliverable**: 5 algorithms ready

---

### Day 4: Experiment Infrastructure (6-8 hours)
- **Morning** (3-4 hrs):
  - [ ] Implement `run_experiment.py`
  - [ ] Create configuration system
  - [ ] Test on small scale (100 episodes)
  
- **Afternoon** (3-4 hrs):
  - [ ] Implement metrics computation
  - [ ] Implement result storage
  - [ ] Test full pipeline

**Deliverable**: Can run full experiments

---

### Day 5: Main Experiments (4-6 hours)
- **Morning** (2-3 hrs):
  - [ ] Run Experiment 1 (Regret Comparison)
  - [ ] Run Experiment 2 (Scaling)
  
- **Afternoon** (2-3 hrs):
  - [ ] Run Experiment 3 (Structure Exploitation)
  - [ ] Run Experiment 4 (Distribution Robustness)
  - [ ] Save all results

**Deliverable**: Main experimental results

---

### Day 6: Analysis & Visualization (6-8 hours)
- **Morning** (3-4 hrs):
  - [ ] Implement visualization functions
  - [ ] Generate main plots (Figures 1-4)
  - [ ] Implement statistical tests
  
- **Afternoon** (3-4 hrs):
  - [ ] Run statistical analysis
  - [ ] Create result tables
  - [ ] Generate publication-ready figures

**Deliverable**: All plots and tables

---

### Day 7: Polish & Documentation (4-6 hours)
- **Morning** (2-3 hrs):
  - [ ] Add docstrings
  - [ ] Format code (black, isort)
  - [ ] Run full test suite
  
- **Afternoon** (2-3 hrs):
  - [ ] Create Colab notebook
  - [ ] Update README with results
  - [ ] Final verification

**Deliverable**: Publication-ready codebase

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Coverage Target**: >80%

**Test Files**:

```python
# tests/test_environment.py
def test_environment_initialization():
    """Test environment initializes correctly."""
    env = BargainingEnvironment(n_issues=3, T_max=20)
    assert env.n_issues == 3
    assert env.T_max == 20

def test_single_negotiation():
    """Test can complete one negotiation."""
    env = BargainingEnvironment()
    opponent = ConcederOpponent()
    
    done = False
    round_count = 0
    while not done and round_count < 20:
        offer = np.array([0.5, 0.3, 0.2])
        response = env.step(offer, opponent)
        done = response['done']
        round_count += 1
    
    assert done or round_count == 20

def test_opponent_concession():
    """Test opponent concedes over time."""
    opponent = ConcederOpponent(alpha=1.5)
    
    offers = []
    for t in range(20):
        offer = opponent.generate_offer(t, [])
        offers.append(offer[0])  # Track first dimension
    
    # Conceder should decrease demand
    assert offers[-1] < offers[0]

# tests/test_algorithms.py
def test_tsb_belief_update():
    """Test TSB updates beliefs correctly."""
    tsb = ThompsonSamplingBargaining(n_types=4)
    initial_alpha = tsb.alpha.copy()
    
    # Simulate outcome
    outcome = {
        'utility': 0.7,
        'type_true': 0,
        'history': []
    }
    tsb.update(outcome)
    
    # Alpha should increase
    assert np.any(tsb.alpha > initial_alpha)

def test_ucb1_selection():
    """Test UCB1 selects correctly."""
    ucb1 = UCB1(n_types=4)
    
    # Initially should explore all types
    selections = []
    for t in range(20):
        sel = ucb1.select_type(t)
        selections.append(sel)
    
    # All types should be selected at least once
    assert len(set(selections)) == 4
```

### 6.2 Integration Tests

```python
# tests/test_integration.py
def test_full_experiment():
    """Test can run complete experiment."""
    config = {
        'T': 100,
        'n_types': 4,
        'opponent_dist': [0.25] * 4,
        'env_params': {'n_issues': 3, 'T_max': 20},
        'alg_params': {}
    }
    
    result = run_single_experiment(
        ThompsonSamplingBargaining,
        config['opponent_dist'],
        config,
        seed=0
    )
    
    assert len(result['regrets']) == 100
    assert all(r >= 0 for r in result['regrets'])

def test_comparison_experiment():
    """Test can compare multiple algorithms."""
    algorithms = [
        ThompsonSamplingBargaining,
        UCB1,
        EpsilonGreedy
    ]
    
    results = {}
    for alg in algorithms:
        result = run_single_experiment(alg, [0.25]*4, config, 0)
        results[alg.__name__] = result
    
    # TSB should have lower regret
    tsb_regret = sum(results['ThompsonSamplingBargaining']['regrets'])
    ucb1_regret = sum(results['UCB1']['regrets'])
    assert tsb_regret < ucb1_regret
```

### 6.3 Regression Tests

Save expected outputs and compare:

```python
def test_regression_exp1():
    """Test Experiment 1 produces expected results."""
    result = run_experiment_1(seed=42)
    
    # Load expected results
    expected = load_expected('exp1_seed42.pkl')
    
    # Compare within tolerance
    np.testing.assert_allclose(
        result['regrets'],
        expected['regrets'],
        rtol=0.01
    )
```

---

## 7. Code Snippets

### 7.1 BargainingEnvironment

```python
# src/environment/bargaining_env.py
import numpy as np
from collections import deque

class BargainingEnvironment:
    """
    Bilateral alternating-offers bargaining environment.
    """
    
    def __init__(self, n_issues=3, T_max=20, delta=0.95, 
                 disagreement_value=0.1):
        """
        Initialize environment.
        
        Args:
            n_issues: Number of negotiation issues
            T_max: Maximum rounds (deadline)
            delta: Discount factor
            disagreement_value: Payoff if no agreement
        """
        self.n_issues = n_issues
        self.T_max = T_max
        self.delta = delta
        self.d = disagreement_value
        
        # Agent utility (fixed)
        self.agent_weights = np.array([0.5, 0.3, 0.2])
        
        self.reset()
    
    def reset(self):
        """Reset for new negotiation."""
        self.round = 0
        self.history = deque(maxlen=self.T_max)
        self.done = False
        self.agreement = None
        return self._get_state()
    
    def step(self, agent_offer, opponent_model):
        """
        Execute one negotiation round.
        
        Args:
            agent_offer: Agent's offer (n_issues,)
            opponent_model: Opponent model to respond
        
        Returns:
            dict with: utility, done, info
        """
        self.round += 1
        
        # Opponent decides to accept or reject
        accept = opponent_model.accept_offer(
            agent_offer, 
            self.round, 
            self.history
        )
        
        if accept:
            # Agreement reached
            utility = self._compute_utility(agent_offer)
            self.done = True
            self.agreement = agent_offer
            
            return {
                'utility': utility,
                'done': True,
                'agreement': agent_offer,
                'round': self.round
            }
        
        # Opponent rejects and makes counter-offer
        if self.round >= self.T_max:
            # Deadline reached, disagreement
            return {
                'utility': self.d,
                'done': True,
                'agreement': None,
                'round': self.round
            }
        
        opponent_offer = opponent_model.generate_offer(
            self.round, 
            self.history
        )
        
        # Agent decides to accept opponent's offer
        opponent_utility = self._compute_utility(opponent_offer)
        accept_threshold = self.d  # Simplified acceptance rule
        
        if opponent_utility >= accept_threshold:
            # Agent accepts
            return {
                'utility': opponent_utility,
                'done': True,
                'agreement': opponent_offer,
                'round': self.round
            }
        
        # Continue negotiation
        self.history.append({
            'round': self.round,
            'agent_offer': agent_offer,
            'opponent_offer': opponent_offer
        })
        
        return {
            'utility': None,
            'done': False,
            'agreement': None,
            'round': self.round
        }
    
    def _compute_utility(self, offer):
        """Compute agent utility for given offer."""
        return np.dot(self.agent_weights, offer)
    
    def _get_state(self):
        """Return current state."""
        return {
            'round': self.round,
            'history': list(self.history),
            'done': self.done
        }
```

### 7.2 OpponentModel

```python
# src/environment/opponent_models.py
import numpy as np

class OpponentModel:
    """Base class for opponent types."""
    
    def __init__(self, utility_weights, name="generic"):
        """
        Args:
            utility_weights: Opponent's utility function weights
            name: Opponent type name
        """
        self.weights = np.array(utility_weights)
        self.name = name
    
    def compute_utility(self, offer):
        """Compute opponent's utility for offer."""
        return np.dot(self.weights, offer)
    
    def generate_offer(self, round, history):
        """Generate offer at given round."""
        raise NotImplementedError
    
    def accept_offer(self, offer, round, history):
        """Decide whether to accept offer."""
        # Logistic acceptance probability
        utility = self.compute_utility(offer)
        reservation = self._get_reservation_utility(round)
        
        prob_accept = 1 / (1 + np.exp(-5 * (utility - reservation)))
        return np.random.random() < prob_accept
    
    def _get_reservation_utility(self, round):
        """Get reservation utility at round."""
        raise NotImplementedError


class ConcederOpponent(OpponentModel):
    """
    Conceder: Decreases demand linearly over time.
    """
    
    def __init__(self, utility_weights=[0.2, 0.5, 0.3], alpha=1.5):
        super().__init__(utility_weights, "conceder")
        self.alpha = alpha
    
    def generate_offer(self, round, history):
        """Concede linearly."""
        t_norm = round / 20  # Normalize to [0, 1]
        concession_factor = (1 - t_norm) ** self.alpha
        
        # Start with maximum demand, concede toward fair split
        max_offer = self.weights / self.weights.sum()
        fair_split = np.ones(len(self.weights)) / len(self.weights)
        
        offer = max_offer * concession_factor + fair_split * (1 - concession_factor)
        return offer / offer.sum()  # Normalize
    
    def _get_reservation_utility(self, round):
        """Reservation decreases over time."""
        t_norm = round / 20
        max_util = self.compute_utility(self.weights / self.weights.sum())
        return max_util * (1 - t_norm) ** self.alpha


class HardlinerOpponent(OpponentModel):
    """
    Hardliner: Maintains high demand until near deadline.
    """
    
    def __init__(self, utility_weights=[0.3, 0.2, 0.5], beta=5):
        super().__init__(utility_weights, "hardliner")
        self.beta = beta
    
    def generate_offer(self, round, history):
        """Hold firm until late."""
        t_norm = round / 20
        
        if t_norm < 0.8:
            # High demand
            offer = self.weights / self.weights.sum()
        else:
            # Sharp concession
            concession = np.exp(-self.beta * (t_norm - 0.8))
            max_offer = self.weights / self.weights.sum()
            fair_split = np.ones(len(self.weights)) / len(self.weights)
            offer = max_offer * concession + fair_split * (1 - concession)
        
        return offer / offer.sum()
    
    def _get_reservation_utility(self, round):
        """High reservation until late."""
        t_norm = round / 20
        max_util = self.compute_utility(self.weights / self.weights.sum())
        
        if t_norm < 0.8:
            return max_util * 0.9
        else:
            return max_util * np.exp(-self.beta * (t_norm - 0.8))
```

---

## 8. Common Pitfalls

### 8.1 Numerical Stability

**Issue**: Likelihood computation can underflow/overflow

**Solution**:
```python
# BAD
likelihood = np.exp(log_likelihood)

# GOOD
log_likelihood = np.clip(log_likelihood, -50, 50)
likelihood = np.exp(log_likelihood)
likelihood = likelihood / (likelihood.sum() + 1e-10)
```

### 8.2 Random Seed Management

**Issue**: Non-reproducible results

**Solution**:
```python
def run_experiment(seed):
    # Set ALL random seeds
    np.random.seed(seed)
    random.seed(seed)
    
    # For libraries
    if torch is not None:
        torch.manual_seed(seed)
```

### 8.3 Memory Leaks

**Issue**: Storing full history grows unbounded

**Solution**:
```python
# Use deque with maxlen
from collections import deque
history = deque(maxlen=T_max)
```

### 8.4 Off-by-One Errors

**Issue**: Round indexing confusion (0-based vs. 1-based)

**Solution**:
```python
# Always use 0-based internally
for round_idx in range(T_max):  # 0 to T_max-1
    # When displaying, add 1
    print(f"Round {round_idx + 1}/{T_max}")
```

---

## 9. Colab Notebook Structure

### 9.1 Main Notebook (`notebooks/full_experiments.ipynb`)

```python
# Cell 1: Setup
!pip install -q numpy scipy matplotlib seaborn pandas tqdm

# Cell 2: Imports
import sys
sys.path.append('/content/regret-optimal-bargaining/src')

from environment.bargaining_env import BargainingEnvironment
from algorithms.thompson_bargaining import ThompsonSamplingBargaining
# ... other imports

# Cell 3: Quick Test
print("Running quick test...")
env = BargainingEnvironment()
tsb = ThompsonSamplingBargaining(n_types=4)
print("✓ All components loaded successfully!")

# Cell 4: Configuration
config = {
    'T': 1000,
    'n_seeds': 10,
    'algorithms': [
        ThompsonSamplingBargaining,
        UCB1,
        EpsilonGreedy,
        FixedStrategy,
        RandomBaseline
    ],
    'opponent_dist': [0.25, 0.25, 0.25, 0.25]
}

# Cell 5: Run Experiments
from experiments.run_all_experiments import run_experiment_suite

print("Running experiments (15-30 minutes)...")
results = run_experiment_suite(config)

# Cell 6: Analysis
from experiments.analysis import analyze_results

summary = analyze_results(results)
print(summary)

# Cell 7: Visualization
from utils.visualization import plot_regret_curves, plot_comparison

plot_regret_curves(results)
plot_comparison(summary)

# Cell 8: Download Results
from google.colab import files
summary.to_csv('results_summary.csv')
files.download('results_summary.csv')
```

---

## 10. Checklist Before Submission

### Code Quality
- [ ] All functions have docstrings
- [ ] Code formatted with `black`
- [ ] Imports organized with `isort`
- [ ] No commented-out code
- [ ] No print statements (use logging)

### Testing
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Regression tests pass
- [ ] Coverage >80%

### Documentation
- [ ] README updated with results
- [ ] TECHNICAL_SPEC accurate
- [ ] EXPERIMENTS protocols followed
- [ ] IMPLEMENTATION guide complete

### Reproducibility
- [ ] Random seeds documented
- [ ] All configs saved
- [ ] Raw results archived
- [ ] Environment pinned (requirements.txt)

### Performance
- [ ] Experiments run in <30 min
- [ ] Memory usage <2GB
- [ ] No obvious bottlenecks

---

**Ready to implement?** Start with Phase 1 (Environment) and work through systematically.

**Questions?** Refer back to TECHNICAL_SPEC.md for mathematical details and EXPERIMENTS.md for protocols.

---

**Version History**:
- v1.0 (2026-02-03): Initial implementation plan for AAMAS 2026 submission
