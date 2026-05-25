# 🌫️ HazeBench: A Novel Dataset for Image and Video Dehazing in Natural Environments

This repository contains the implementation of haze removal and visibility enhancement techniques applied to a diverse real-world dataset. The dataset is designed to evaluate performance under varying haze conditions across multiple environments.

---

## 📂 Dataset

The dataset is publicly available on Zenodo:

🔗 https://doi.org/10.5281/zenodo.19678785

### Dataset Structure

The dataset is organized into three main folders:

1. **Original Videos**  
   Contains raw videos captured under different haze conditions.

2. **Patch Videos**  
   Short video clips (1–15 seconds) extracted from original videos.

3. **Images**  
   Frames extracted from videos for detailed analysis.

Each folder includes the following **five categories**:
- Indoor (semi-outdoor environments like lawns and courtyards)
- Mountains (high-altitude haze conditions)
- Night (low-light haze with artificial illumination)
- Road (traffic and driving environments)
- Rural Areas (open countryside scenes)

---

## ⚙️ Methodology

The proposed framework applies image and video processing techniques for haze removal and visibility enhancement. The method focuses on improving contrast, restoring scene details, and handling diverse real-world haze conditions across spatial and temporal variations.

---

## 📁 Code Structure

This repository includes three main components:

- **Existing Model**: Baseline haze removal method used for comparison.
- **Proposed Method (Final Output)**: Improved haze removal approach for enhanced visibility.
- **Fusion Model**: Combines outputs of both models to improve visual quality and robustness.

---

## 📊 Results

The results demonstrate improved visibility and enhanced scene clarity across different haze conditions, including dense fog, nighttime haze, and outdoor environmental variations.

---

## 🔁 Reproducibility

All code, dataset links, and implementation details are publicly available to ensure reproducibility. The dataset is hosted on Zenodo, and the full implementation is available in this repository.

---

## 🚀 Applications

- Autonomous navigation  
- Surveillance systems  
- Traffic monitoring  
- Environmental analysis  

---

## 📄 Citation

If you use this dataset or code, please cite:

**HazeBench: A Novel Dataset for Image and Video Dehazing in Natural Environments**  
Manuscript submitted to *The Visual Computer*, Springer Nature.

### GitHub Repository
https://github.com/YOUR-USERNAME/HazeBench

### Dataset (Zenodo)
https://zenodo.org/records/19678785

---

## 🤝 Acknowledgements

We thank the research community for providing benchmark datasets that inspired this work.
