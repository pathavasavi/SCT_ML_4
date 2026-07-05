# SkillCraft Technology Machine Learning Internship - Task 4

## 🖐️ Hand Gesture Recognition Using Convolutional Neural Networks (CNN)

### 📌 Project Overview
This project develops a Deep Learning computer vision model using a Convolutional Neural Network (CNN) to recognize and classify distinct hand gestures. Built using TensorFlow and Keras, the system processes RGB images of hand gestures, tracks spatial feature patterns, and classifies them across dynamic action categories.

### 📊 Dataset Structure
The system is built and trained using the **[Hand Gestures Dataset on Kaggle](https://www.kaggle.com/datasets/parikshitkumar/hand-gestures)**.
* **Format:** Multi-class RGB (color) images.
* **Organization:** Managed locally in an `images/` directory split into unique gesture class subfolders.

> ⚠️ **Note:** The local image dataset directory is excluded from remote version tracking via `.gitignore` to comply with standard software hosting size constraints.

### ⚙️ Deep Learning Pipeline & Architecture
1. **Data Preprocessing & Augmentation:** Automatically scans categorical directories with `opencv-python` (`cv2`), standardizes all inputs to \((64 \times 64)\) resolution, scales feature values to a normalized \([0.0, 1.0]\) floating-point range, and applies one-hot encoding via Keras utilities.
2. **Train-Test Split:** Implements an 80/20 train-test validation partition strategy using `train_test_split`.
3. **CNN Network Architecture Sequential Design:**
   * **Input Layer:** Accepts \((64, 64, 3)\) shape matrices.
   * **Convolutional Layer 1:** 32 filters, \((3 \times 3)\) kernel size, ReLU activation.
   * **Max Pooling Layer 1:** \((2 \times 2)\) pooling window downsampling spatial dimensions.
   * **Convolutional Layer 2:** 64 filters, \((3 \times 3)\) kernel size, ReLU activation.
   * **Max Pooling Layer 2:** \((2 \times 2)\) pooling window downsampling spatial dimensions.
   * **Flattening Layer:** Reshapes 2D matrices into 1D feature arrays.
   * **Dense Layer (Hidden):** 128 structural units using ReLU activation.
   * **Regularization (Dropout):** 50% neural drop probability constraint preventing overfitting.
   * **Output Layer:** Dynamically calculates soft-max categorical distribution metrics matched directly to individual gesture class counts.
4. **Optimization Pipeline:** Uses the `Adam` optimizer, tracking performance via `categorical_crossentropy` losses over 10 training epochs.
5. **Analytics Reporting:** Generates distinct `matplotlib` visualization figures plotting performance trajectories for validation accuracy and training loss.

### 🛠️ Local Installation & Setup

#### 1. Clone the Repository
```bash
git clone https://github.com
cd your-repository-name
```

#### 2. Install Required Dependencies
Ensure you have Python installed, then run the environment mapping:
```bash
pip install numpy opencv-python tensorflow scikit-learn matplotlib notebook
```

#### 3. Dataset Configuration
1. Download the archive bundle directly from the [Kaggle Hand Gestures Dataset Page](https://www.kaggle.com/datasets/parikshitkumar/hand-gestures).
2. Unzip the target file array and structure your project directory locally as follows:
   ```text
   📁 your-repository-name
   ├── 📁 images
   │   ├── 📁 1-palm
   │   ├── 📁 2-L
   │   └── 📁 ... (all gesture folders)
   └── 📄 Task4.ipynb
   ```

#### 4. Execution
Launch your local Jupyter interface to run the script:
```bash
jupyter notebook Task4.ipynb
```
