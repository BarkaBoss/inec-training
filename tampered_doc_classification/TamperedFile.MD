markdown_content = """# Workshop Guide: Form Tampering Classifier with PyTorch and Streamlit

Welcome to the hands-on workshop guide! In this tutorial, we will break down a Python application that uses a Deep Learning model to determine whether a document/form image is **Authentic** or **Tampered**. The application features an interactive UI built with **Streamlit**, allowing users to train a neural network and test live images directly in their browser.

---

## Table of Contents
1. [Prerequisites & Directory Setup](#1-prerequisites--directory-setup)
2. [Section 1: Library Imports & Configuration](#2-section-1-library-imports--configuration)
3. [Section 2: Data Preprocessing & Augmentation](#3-section-2-data-preprocessing--augmentation)
4. [Section 3: Defining the Deep Learning Model (Transfer Learning)](#4-section-3-defining-the-deep-learning-model-transfer-learning)
5. [Section 4: The Streamlit Interface & Training Loop](#5-section-4-the-streamlit-interface--training-loop)
6. [Section 5: Inference / Testing Live Documents](#6-section-5-inference--testing-live-documents)
7. [How to Run the Application](#7-how-to-run-the-application)

---

## 1. Prerequisites & Directory Setup

Before running the code, ensure you have the required packages installed. You can install them via pip: