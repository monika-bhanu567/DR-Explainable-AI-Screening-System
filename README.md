# Explainable AI System for Diabetic Retinopathy Screening

An AI-assisted diabetic retinopathy screening system that combines retinal image classification, explainability, uncertainty estimation, image-quality assessment, safety gating, and doctor-in-the-loop decision support.

> **Screen → Explain → Prioritize → Refer → Follow-up → Monitor**

---

## 🚨 Problem

Diabetic Retinopathy (DR) is a diabetes-related retinal disease that can lead to vision loss if it is not detected and managed in time.

AI-based retinal image classification can assist screening, but a simple prediction is not sufficient for healthcare-oriented decision support.

A useful screening system should also answer:

- What DR severity does the model predict?
- How reliable is the prediction?
- Which image regions influenced the prediction?
- Is the retinal image of sufficient quality?
- Should the AI result be trusted automatically?
- When should a doctor review the case?
- Can previous screening results be compared?

This project addresses these requirements through an explainable and safety-aware AI screening pipeline.

---

## 🎯 Objectives

- Classify retinal fundus images into five DR severity grades.
- Preserve retinal image information using higher-resolution input.
- Handle class imbalance during training.
- Calibrate model confidence.
- Estimate prediction uncertainty.
- Generate visual explanations using Grad-CAM++.
- Assess retinal image quality before AI screening.
- Prevent autonomous decisions when model reliability is low.
- Support doctor-in-the-loop review.
- Maintain disagreement/audit information.
- Support bilateral consistency analysis.
- Provide prototype macular/exudate proximity analysis.
- Support longitudinal progression comparison.

---

## 🧠 AI/ML Pipeline

```text
Retinal Fundus Image
        │
        ▼
Image Preprocessing
        │
        ▼
Image Quality Assessment
        │
        ▼
EfficientNet-B0
512 × 512 Input
        │
        ▼
5-Class DR Prediction
        │
        ├──────────────► Temperature Scaling
        │                       │
        │                       ▼
        │                 Calibrated Score
        │
        ├──────────────► Entropy
        │                       │
        │                       ▼
        │                  Reliability
        │
        ▼
Grad-CAM++ Explanation
        │
        ▼
Safety Gate
        │
        ▼
Smart Triage
        │
        ▼
Doctor Review
        │
        ▼
Referral / Follow-up
        │
        ▼
Longitudinal Monitoring