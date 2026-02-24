# Semantic Opinion Dynamics: LLM Agents on Complex Networks

**Replaces 50 years of weighted-average opinion dynamics with in-context learning.**

This project implements the research idea from your assignment: simulating opinion dynamics using Large Language Model agents on social networks. Instead of classical DeGroot models where opinions are scalars averaged numerically, we model agents as LLMs with text-based beliefs that update through conversation.

---

## 🎯 Project Overview

**Research Question:**  
Can LLM agents capture semantic nuances of polarization (framing, rhetoric, logical fallacies) that classical opinion dynamics miss?

**Key Innovation:**  
- **Classical approach:** Opinions are numbers in [0,1], update = weighted average of neighbors
- **Our approach:** Opinions are text, update = LLM reads neighbors' texts and generates new opinion in-context

**Experiments:**
1. **Baseline:** Track semantic variance over time using SBERT embeddings (completed)
2. **DeGroot Comparison:** Show that LLMs can maintain polarization where DeGroot converges
3. **Bot Intervention:** Measure network resilience to disinformation
4. **Topology Study:** Compare Scale-free vs. Small-world vs. Random networks

---

## 📁 Project Structure

```
├── config.py                    # Configuration & persona definitions
├── network_generation.py        # Graph creation & visualization
├── persona_assignment.py        # Assign personas to nodes
├── simulation.py               # Core LLM simulation engine
├── measurement.py              # Semantic embedding analysis
├── main.py                     # Orchestration script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key


```bash
# For Anthropic (Claude)
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Run Baseline Simulation

```bash
python main.py --mode baseline
```

This will:
- Create a Karate Club network (34 nodes)
- Assign diverse personas
- Run 8 rounds of opinion dynamics
- Generate semantic variance plots
- Save results to `/outputs/`

---

## 🧪 Experiment Modes (Completed)


### 1. Baseline Simulation
```bash
python main.py --mode baseline
```
**Outputs:**
- `network_structure.png` - Network visualization with persona colors
![Network Structure](outputs/network_structure.png)

- `semantic_variance.png` - Variance over time
![Semantic Variance](outputs/semantic_variance.png)

- `sample_opinions.txt` - Opinion trajectories for 3 agents

### 2. Bot Intervention Study
```bash
python main.py --mode intervention
```
Tests network resilience by adding a high-degree "disinformation bot" node.
**Outputs:**
- `intervention_comparison.png` - Intervention Comparison
![Intervention Comparison](outputs/intervention_comparison.png)


### 3. Topology Comparison
```bash
python main.py --mode comparison
```
Compares Scale-free, Small-world, and Random networks.
**Outputs:**
- `topology_comparison.png` - Topology Comparison
![Topology Comparison](outputs/topology_comparison.png)



### 4. DeGroot Comparison
```bash
python main.py --mode degroot
```
Compares LLM semantic dynamics with classical DeGroot model.
**Outputs:**
- `llm_vs_degroot.png` - LLM VS Degroot Comparison
![Degroot Comparison](outputs/llm_vs_degroot.png)


---

## 🎭 Persona Design

We define 6 persona archetypes on the "AI Regulation" controversy:

| Archetype | Description | Example |
|-----------|-------------|---------|
| **Strong Pro** | Safety-focused, urgent regulation needed | Tech ethicist, AI safety researcher |
| **Moderate Pro** | Supports targeted regulation, worries about overreach | Software engineer, professor |
| **Centrist** | Genuinely undecided, highly persuadable | Policy analyst, journalist |
| **Moderate Anti** | Worries regulation will hurt small players | Startup founder, open-source advocate |
| **Strong Anti** | Free markets only, government is incompetent | Libertarian, accelerationist |
| **Contrarian** | Reflexively opposes consensus | Devil's advocate |

Each persona includes:
- **Background & Values:** Grounds the agent's perspective
- **Reasoning Style:** Evidence-based, emotional, ideological, etc.
- **Persuadability:** How easily swayed by neighbors

---

## 📊 Measurement Methodology

### Semantic Variance
1. Encode all opinion texts using SBERT (`all-MiniLM-L6-v2`)
2. Compute pairwise cosine distances between embeddings
3. **Variance = Mean(pairwise distances)**

**Interpretation:**
- **Increasing variance** → Polarization (agents diverging)
- **Decreasing variance** → Convergence (agents agreeing)

### DeGroot Baseline
Map personas to scalars:
- Strong Pro: 0.9
- Moderate Pro: 0.65
- Centrist: 0.5
- Moderate Anti: 0.35
- Strong Anti: 0.1

Update rule: `opinion[t+1] = mean(neighbors' opinions[t])`

DeGroot **always converges** to consensus. Our LLM agents may not!

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# API Settings
API_PROVIDER = "anthropic"  # or "openai"
API_MODEL = "claude-sonnet-4-20250514"

# Network Settings
NETWORK_SIZE = 30
NETWORK_TYPE = "karate"  # or "scale_free", "small_world", "random"
SIMULATION_ROUNDS = 8

# Topic
CONTROVERSIAL_TOPIC = "AI Regulation"
```

---


