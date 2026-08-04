## Reading Path

If you're new to this work, I recommend starting with the essays in order:

1. Essay #1 — The Fog of Certainty: On Deep Learning's Secret Vice (Published in Medium)
2. Essay #2 — The Attack Landscape: Why Silence Is the Vulnerability (Published in Towards AI)
3. Essay #3 — Walls, Shields and Illusions: Defenses and their limits(Published in Towards AI)
4. The bridge essay — The Bridge: From Resistance to Awareness, Why Defenses Alone Will Never Be Enough (Published in Medium)
5. Essay #4 _ Learning to Say “I Don’t Know”: The First Floor of Awareness (Published in Towards AI)
6. Essay #5 _ The Geometry of Fragility: Feeling the Decision Boundary (Published in Medium)
7. Essay #6 _ (Coming soon...)
8. ...

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

📖 Article:
Read on Medium: https://medium.com/@maedeh.torkian/the-fog-of-certainty-on-deep-learnings-secret-vice-ed73d6ac3242

**Notebook(s):**
* No companion notebook for this essay.

---

### Essay #2 — *The Attack Landscape: Why Silence Is the Vulnerability*

A study of adversarial behavior beyond visible failures, asking whether omission, selective behavior, and silence can themselves become vulnerabilities.

📖 Article:
Read on Towards AI: https://pub.towardsai.net/the-attack-landscape-why-silence-is-the-vulnerability-f27dabfccb9b

**Notebook(s):**

* (https://github.com/Maee127/Adversarial-Notebooks/blob/master/fgsm_adversarial_notebook.ipynb)

---

### Essay #3 — *Walls, Shields and Illusions: Defenses and their limits* 

An exploration of adversarial training, defensive distillation, gradient masking, and a deeper question:

**Can any defense remain unbreakable given enough computational power and knowledge of the model?** 
If every wall eventually breaks, then the entire framework of defense is built on sand.

📖 Article:
Read on Towards AI: https://medium.com/towards-artificial-intelligence/walls-shields-and-illusions-defenses-and-their-limits-cd67b6ec4f92

**Notebook(s):**

* (https://github.com/Maee127/Adversarial-Notebooks/blob/master/%233/Note02_Adversarial_Training_PGD.ipynb)

---

### Essay bridge —  *The Bridge: From Resistance to Awareness* 

**Why Defenses Alone Will Never Be Enough**

The bridge asks a different question.
Instead of asking:

    How can we build a stronger wall?

it asks:

    What if resistance alone is the wrong objective?

This essay connects the study of adversarial robustness with a broader perspective: model awareness.
It is not the conclusion of the first part of the journey.

It is the path leading to the next one(Essay #4).

📖 Article:
Read on Medium: https://medium.com/@maedeh.torkian/the-bridge-from-resistance-to-awareness-58ea4156ad3f

**Notebook(s):**
* No companion notebook for this essay.
---

### Essay #4 — *Learning to Say “I Don’t Know”: The First Floor of Awareness*

We built the voice.

A decision gate — a simple, auditable rule that says: “If uncertainty is high, do not predict. Ask for help.”
This is not a new wall. 

> **It is a voice, and it is the first floor of a structure this series will keep building on.**

📖 Article:
Read on Towards AI: https://pub.towardsai.net/learning-to-say-i-dont-know-the-first-floor-of-awareness-1a9faae389c0

**Notebook(s):**

* (https://github.com/Maee127/Adversarial-Notebooks/blob/master/%234/Notebook_%234.ipynb)


---
### Essay #5 — *The Geometry of Fragility: Feeling the Decision Boundary*

We gave the model a voice — a way to say: “I am not sure” and “this input looks strange.

It worked — on natural data. But it failed — on adversarial data.

**That is the blind spot we must now close.**

The model needs to be aware of its own geometry — not just the shape of the data distribution, but the shape of its decision boundary. 

📖 Article:
Read on Medium: https://medium.com/@maedeh.torkian/the-geometry-of-fragility-feeling-the-decision-boundary-2d1c50f402ee?sharedUserId=maedeh.torkian


**Notebook(s):**

* (https://github.com/Maee127/Adversarial-Notebooks/blob/master/%235/Notebook_5.ipynb)

---
### Essay #6 (in progress...)

---
## Repository Structure

```text
/
├── essay_01/
├── essay_02/
├── essay_03/
├── essay_bridge/
├── essay_04/
├── essay_05/
├── essay_06/
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

> *“Every honest question builds the next bridge.”*
