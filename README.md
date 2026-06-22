## Reading Path

If you're new to this work, I recommend starting with the essays in order:

1. Essay #1 — The Fog of Certainty: On Deep Learning's Secret Vice (Published in Medium)
2. Essay #2 — The Attack Landscape: Why Silence Is the Vulnerability (Published in Towards AI)
3. Essay #3 — Walls, Shields and Illusions: Defenses and their limits(Published in Towards AI)
4. The bridge essay — Why Defenses Alone Will Never Be Enough(coming soon)
5. ...

# Adversarial AI: Attacks, Defenses, and Model Awareness

> *The attacker adapts. The wall crumbles.
> The researcher adapts. The question changes.*

This repository accompanies an ongoing and evolving study of adversarial attacks, defenses, and model behavior — following questions as they change.

It began with a simple question:

> **How are we vulnerable?**

From there, the investigation moved through adversarial perturbations, model behavior, silence as vulnerability, and the changing relationship between attacks and defenses.

The goal of this repository is not only to reproduce results, but to document the process of inquiry — including experiments, failures, changing assumptions, and the unexpected directions that emerge while working with models.

Each essay is accompanied by notebooks, code, and supporting material.

## Series Roadmap

### Essay #1 — *The Fog of Certainty: On Deep Learning's Secret Vice*

The starting point of this series.

This essay began with a shift in framing:

Not “Are we vulnerable?” — but “How are we vulnerable?”

An exploration of adversarial vulnerability, assumptions in model robustness, and the foundations of the questions explored throughout the series.

Article:
Read on Medium: https://medium.com/@maede.torkian/the-fog-of-certainty-on-deep-learnings-secret-vice-ed73d6ac3242

**Notebook(s):**
* No companion notebook for this essay.

---

### Essay #2 — *The Attack Landscape: Why Silence Is the Vulnerability*

A study of adversarial behavior beyond visible failures, asking whether omission, selective behavior, and silence can themselves become vulnerabilities.

Article:
Read on Towards AI: https://pub.towardsai.net/the-attack-landscape-why-silence-is-the-vulnerability-f27dabfccb9b

**Notebook(s):**

* (https://github.com/Maee127/Adversarial-Notebooks/blob/master/fgsm_adversarial_notebook.ipynb)

---

### Essay #3 — *Walls, Shields and Illusions: Defenses and their limits* 

An exploration of adversarial training, defensive distillation, gradient masking, and a deeper question:

**Can any defense remain unbreakable given enough computational power and knowledge of the model?** 
If every wall eventually breaks, then the entire framework of defense is built on sand.

Article:
Read on Towards AI: https://medium.com/towards-artificial-intelligence/walls-shields-and-illusions-defenses-and-their-limits-cd67b6ec4f92

**Notebook(s):**

* (https://github.com/Maee127/Adversarial-Notebooks/blob/master/%233/Note02_Adversarial_Training_PGD.ipynb)

---

### Essay bridge —  *The bridge essay (between #3 and #4)* (working title)

**What if we stopped playing the attacker-defender game?**

---
## Repository Structure

```text
/
├── essay_01/
├── essay_02/
├── essay_03/
├── essay_bridge/
├── notebooks/
└── README.md
```

## A Note on This Work

This repository evolves alongside the essays.

Some conclusions may change.
Some assumptions may fail.
Some questions may become more interesting than their answers.

That is part of the process.

> *“The road has been changed in every movement.”*
