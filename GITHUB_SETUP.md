# GitHub Repository Setup Instructions

## Quick Setup (5 minutes)

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `regret-optimal-bargaining`
3. Description: `Regret-Optimal Exploration in Repeated Alternating-Offers Bargaining (AAMAS 2026)`
4. Visibility: **Public** (for open science)
5. **DO NOT** initialize with README (we already have one)
6. Click "Create repository"

### Step 2: Push to GitHub

```bash
cd regret-optimal-bargaining

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/regret-optimal-bargaining.git

# Push
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

###  Step 3: Verify

Go to `https://github.com/YOUR_USERNAME/regret-optimal-bargaining`

You should see:
- ✅ README.md (with badges and overview)
- ✅ TECHNICAL_SPEC.md (54KB - complete mathematical formulation)
- ✅ EXPERIMENTS.md (20KB - experimental protocols)
- ✅ LICENSE (MIT)
- ✅ requirements.txt
- ✅ .gitignore

---

## Repository Settings (Recommended)

### Enable GitHub Pages (for project website)

1. Go to Settings → Pages
2. Source: Deploy from branch `main`, folder `/docs` (or root)
3. Save

### Add Topics

Go to repository home → ⚙️ (next to About) → Add topics:
- `automated-negotiation`
- `multi-agent-systems`
- `thompson-sampling`
- `regret-minimization`
- `aamas2026`
- `bargaining`
- `reinforcement-learning`

### Add Description

```
Regret-Optimal Exploration in Repeated Alternating-Offers Bargaining - Thompson Sampling for Bargaining (TSB) with O(√T/B) regret guarantees. AAMAS 2026 submission.
```

---

## What's Included

| File | Size | Contents |
|------|------|----------|
| **README.md** | 10KB | Project overview, quick start, results preview |
| **TECHNICAL_SPEC.md** | 24KB | Complete math: problem formulation, algorithms, proofs, regret bounds |
| **EXPERIMENTS.md** | 20KB | 9 experiments, ablations, metrics, statistical protocols |
| **LICENSE** | 1KB | MIT License |
| **requirements.txt** | 118B | Python dependencies |
| **.gitignore** | 500B | Ignore patterns |

**Total**: ~55KB of publication-ready technical documentation

---

## Next Steps (Implementation)

### Immediate (Week 1)
1. Create `src/` directory structure
2. Implement core algorithms (TSB, UCB1, etc.)
3. Implement bargaining environment
4. Write unit tests

### Short-term (Week 2)
5. Run main experiments (Exp 1-4)
6. Generate plots
7. Draft paper sections 4-5 (experiments, results)

### Medium-term (Week 3)
8. Ablation studies
9. Robustness analysis
10. Finalize paper draft

---

## Collaboration Workflow

### For Contributors

```bash
# Fork repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/regret-optimal-bargaining.git
cd regret-optimal-bargaining

# Create feature branch
git checkout -b feature/my-improvement

# Make changes, commit
git add .
git commit -m "Add: my improvement"

# Push to your fork
git push origin feature/my-improvement

# Open Pull Request on GitHub
```

### Code Review Checklist
- [ ] Tests pass
- [ ] Code follows PEP 8
- [ ] Docstrings added
- [ ] Results reproducible

---

## Publication Workflow

### Paper Draft (Google Docs or Overleaf)
1. Share with co-authors
2. Iterate on writing
3. Incorporate reviewer feedback

### Code Freeze (Pre-Submission)
1. Tag release: `v1.0-aamas2026-submission`
   ```bash
   git tag -a v1.0-aamas2026-submission -m "Code snapshot for AAMAS 2026 submission"
   git push origin v1.0-aamas2026-submission
   ```

2. Archive on Zenodo for DOI
   - Go to https://zenodo.org/
   - Link GitHub repo
   - Generate DOI

3. Update paper with DOI and GitHub link

---

## Troubleshooting

### Issue: Push rejected

**Cause**: Remote has changes you don't have locally.

**Solution**:
```bash
git pull origin main --rebase
git push origin main
```

### Issue: Large files rejected

**Cause**: File >100MB (GitHub limit).

**Solution**: Use Git LFS or store in external location (Google Drive, OSF.io).

---

## Repository Maintenance

### Keep README Updated
- Update results as experiments complete
- Add publication status
- Update citation once published

### Version Tags
- `v0.1.0` - Initial specification
- `v0.2.0` - Core implementation
- `v1.0.0` - AAMAS submission
- `v1.1.0` - Camera-ready (if accepted)

### Archive After Publication
- Add "Published at AAMAS 2026" badge
- Link to paper PDF
- Update citation with DOI

---

**Questions?** Open an issue on GitHub!
