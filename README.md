# DeepCare: Predicting Patient Drop-off Using Big Data

**Course**: CSL7110 - Machine Learning with Big Data  
**Institute**: IIT Jodhpur  
**Team**: Vishal Dangiwala, Hariom Birla, Himanshu Pathak  
**Instructor**: Dr. Dip Sankar Banerjee

---

##  Overview
DeepCare is a data mining and machine learning project designed to predict patient drop-off using large-scale hospital data. We integrated and processed multiple datasets (demographics, visits, logs), engineered features, and trained ML models to classify patients by dropout risk.

---

##  Tech Stack
- Python 3.11
- Apache Spark (PySpark)
- Google Colab
- Power BI
- Hadoop (for XML and CSV parsing)

---

## 📊 Dataset
Three anonymized datasets:
- **Demographics**: Age, gender, income, etc.
- **Visits**: Appointment type, satisfaction, cost, etc.
- **Logs**: Event-level medical logs and timestamps

---

##  Machine Learning
- Clustering: KMeans (3 segments)
- Models: Logistic Regression, Random Forest, Gradient Boosted Trees
- Best Model: GBT (AUC: 0.55)

---

##  Results
- Segment 1 had the highest dropout rate.
- Drop-off prediction is moderately effective with room for optimization.
- Visualizations built with Power BI for drop-off insights.

---

##  Future Scope
- Incorporate time-series features and deep learning.
- Real-world deployment for continuous learning and feedback.
- Better intervention strategies to retain high-risk patients.

---

##  File Notice
Datasets are large; upload was limited on GitHub. You can access the full datasets via the Google Drive link provided in the txt file.

---
