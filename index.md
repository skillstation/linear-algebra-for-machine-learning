# Linear Algebra for Machine Learning

(With special emphasis on optical charecter recognition)

## Course Title:
**Linear Algebra for Machine Learning**

---

## Course Outcomes (COs)
By the end of this course, learners will be able to:

1. **Represent images and datasets using vector spaces** and perform fundamental vector/matrix operations relevant to machine learning.
2. **Apply matrix transformations** for preprocessing, noise reduction, and feature extraction in image-based tasks.
3. **Implement dimensionality reduction techniques** (PCA, SVD) to compress and analyze high-dimensional image data.
4. **Use orthogonality, projections, and eigen-decomposition** to derive meaningful features from handwritten characters.
5. **Formulate and solve linear models** using systems of equations and optimization principles.
6. **Build a complete Python-based digit recognition system**, integrating linear algebra concepts with machine learning workflows.
7. **Interpret the role of linear algebra in advanced ML architectures**, including neural networks and convolution operations.

---

## Softwares Used

• MATLAB
• Python

---

## Course Outline (12–14 Weeks)

### Module 1 – Introduction to Linear Algebra for ML
- Why ML needs Linear Algebra
- Images as matrices and vectors
- Basics of NumPy for linear algebra
**Lab:** Convert PNG digit (0–9) into grayscale matrices & vectors.

### Module 2 – Vectors, Vector Spaces & Inner Products
- Vector representation
- Basis, dimension, linear independence
- Dot product, norms, distances
**Lab:** Distance-based digit classifier (nearest vector matching).

### Module 3 – Matrices & Linear Transformations
- Matrix operations, rank, invertibility
- Image scaling, rotation, shearing as matrix transforms
- Linear transformations for preprocessing
**Lab:** Apply transformations on handwritten digits.

### Module 4 – Systems of Linear Equations & Least Squares
- Ax = b formulation
- Gaussian elimination
- Least squares minimization
**Lab:** Fit a simple linear classifier using least squares.

### Module 5 – Eigenvalues, Eigenvectors & Diagonalization
- Geometric meaning of eigenvectors
- Covariance matrices and variance
- Spectral decomposition
**Lab:** Compute eigenvectors of digit datasets.

### Module 6 – Orthogonality & Projections
- Orthogonal/Orthonormal bases
- Projection of data onto subspaces
- Gram–Schmidt process
**Lab:** Project digit images to low-dimensional subspaces.

### Module 7 – Principal Component Analysis (PCA)
- Deriving PCA from eigen-decomposition
- PCA for image compression
- Dimensionality reduction pipeline
**Lab:** Build PCA for “eigen-digit” extraction and visualization.

### Module 8 – Singular Value Decomposition (SVD)
- SVD as a generalized PCA
- Low-rank matrix approximations
- Noise filtering
**Lab:** Reconstruct digits using top-k singular values.

### Module 9 – Optimization & Gradient-Based Methods
- Gradient descent in vector spaces
- Linear and logistic regression foundations
- Cost functions
**Lab:** Implement gradient descent to classify digits (binary or multi-class).

### Module 10 – Convolution as a Linear Operator
- Convolution = matrix multiplication
- Filters as kernels
- Feature extraction via edges, corners, strokes
**Lab:** Write your own convolution function in Python.

### Module 11 – Basics of Neural Networks (Linear Algebra View)
- Weight matrices, bias vectors
- Activation functions
- Forward/backward propagation as matrix operations
**Lab:** Build a simple 1-hidden-layer neural network using NumPy (no frameworks).

### Module 12 – End-to-End Character Recognition System
- Dataset preparation
- Feature engineering pipeline
- Training + performance evaluation
- Error analysis
**Lab:** Complete integration and testing of the digit-recognition model.

### Module 13 – Extensions to Real-World OCR (Optional)
- Multi-layer CNNs (theoretical overview)
- Scanning and segmenting handwritten documents
- Model deployment (Flask/Streamlit)

### Module 14 – Project Presentation & Review
- Demonstration
- Report submission
- Reflection on mathematical foundations

---

## Final Project: Character Recognition Using Linear Algebra

**Project Title:** *EigenDigits: A Linear Algebra–Powered Handwritten Character Recognition System*

### Project Abstract
This project aims to develop a complete handwritten character recognition system using foundational concepts of linear algebra. Starting with a small dataset of digit images (0–9), each image is represented as a high-dimensional vector. The system performs preprocessing using matrix transformations, followed by dimensionality reduction through Principal Component Analysis (PCA) and Singular Value Decomposition (SVD). Orthogonality and projections are used to extract dominant features from the digit images, forming a reduced feature space known as the “eigen-digit” basis. A linear classifier is then trained using least squares and gradient descent techniques. The final model predicts the label of any handwritten digit by comparing its projection in the feature space to the learned representations. This project demonstrates how vector spaces, eigenvalues, SVD, and optimization form the mathematical backbone of modern machine learning and computer vision.

---

## Instructor Notes & Deliverables

- Weekly lecture slides and code notebooks (NumPy-only for clarity) recommended.
  
- Datasets: start with 10 PNGs (0–9) for initial labs; scale to MNIST for full project.
  
- Assessment: weekly labs (40%), midterm project checkpoint (20%), final project (40%).

---

**Instructor:** Elangovan M. G., IITM

**Course length:** 12–14 weeks

**Primary tools:** Python (NumPy, Matplotlib), optional: scikit-learn, PyTorch/TensorFlow for extensions
