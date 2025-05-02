# 🚶‍♀️ pm4mobility — Process Mining Experiments on Urban Mobility Data

**pm4mobility** is an experimental repository for applying **traditional** and **object-centric process mining (OCPM)** techniques to urban mobility data. The project is part of a research study exploring how event log abstractions can model individual and multimodal travel behavior.

---

## 🎯 Purpose

- Transform real-world mobility data (e.g. triplegs or OD flows) into process mining event logs
- Apply and compare traditional process mining and object-centric process mining (OCPM)
- Explore how case-based and object-centric abstractions affect process interpretation

---

## 📘 Overview

The project includes two experimental notebooks:

- `01_traditional_pm.ipynb`: Applies traditional process mining using the Heuristic Miner and Petri net models.
- `02_ocel_pm.ipynb`: Applies object-centric process mining using directly-follows graphs (DFGs).

The resulting models and visualizations are saved in the `figures/` directory.

---

## 🗂️ Data

The original data is based on **Call Detail Records (CDRs)** that were transformed into:

1. **Triplegs** — semantically enriched segments of continuous movement
2. **Origin–Destination (OD) flows** — derived from triplegs and used for process abstraction

> ⚠️ **Note**: Due to privacy restrictions, the data is not publicly available.  
> However, the data structures and intermediate steps are documented and visible in the notebooks.

---

## ⚙️ Status

This is an experimental research repository, not a finished Python package. All logic and results are explored within the notebooks.

---

## 📄 License

MIT License © Khristina Filonchik, 2025

