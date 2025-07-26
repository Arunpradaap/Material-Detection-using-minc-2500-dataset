# 🧠 Real-Time Material Classification Using Deep Learning

This project implements a real-time material classification system using deep learning and computer vision. It uses a webcam to capture object images and a fine-tuned **ResNet18** model to identify the **material type** from a set of 23 common categories.

---

## 📁 Project Structure

- `material_classifier.py` – Real-time webcam-based material detection
- `train_material_classifier.py` – Training script for material classifier
- `organize_minc_dataset.py` – Utility to organize the MINC-2500 dataset
- `models/material_classifier_resnet18.pt` – Pretrained PyTorch model (trained on MINC-2500)
- `requirements.txt` – Python dependencies
- `README.md` – Project overview (this file)

---

## 🔧 Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Arunpradaap S/Material-Detection-using-minc-2500-datasetMaterial-Detection-using-minc-2500-dataset.git
   cd yMaterial-Detection-using-minc-2500-datasetMaterial-Detection-using-minc-2500-dataset

Install dependencies

bash

pip install -r requirements.txt
Ensure model file is present
Place the pretrained material_classifier_resnet18.pt file in the models/ directory.

You can train a new model using train_material_classifier.py.

Run the live material detector

bash
Copy
Edit
python material_classifier.py
🧠 Model Details
Architecture: ResNet18

Training Data: MINC-2500 Dataset

Classes: 23 material types

This model was trained using the original MINC-2500 dataset, developed by Bell et al. at Cornell and Adobe.
Citation:

nginx
Copy
Edit
Sean Bell, Paul Upchurch, Noah Snavely, Kavita Bala. "Material Recognition in the Wild with the Materials in Context Database." CVPR 2015.
Supported Material Classes:
Copy
Edit
aluminum_foil, asphalt, brick, cardboard, carpet, ceramic,
concrete, fabric, foliage, food, glass, grass, gravel,
hair, leather, metal, mirror, paper, plastic, polished_wood,
soil, stone, wood
📂 Dataset Used
MINC-2500 (Materials in Context)
Original source: http://opensurfaces.cs.cornell.edu/publications/minc/
Download via: https://www.kaggle.com/datasets/kmader/materials-data

📸 Example Output
The webcam feed displays:

makefile
Copy
Edit
Material: glass
FPS: 24.18
👤 Author
Arunpradaap S
GitHub   :https://github.com/Arunpradaap
LinkedIn :https://www.linkedin.com/in/arunpradaap

📄 License
MIT License — Free to use, modify, and distribute with credit to original dataset authors and this project.

## 📄 License

MIT License — Free to use, modify, and distribute with credit to:
- The original authors of the MINC-2500 dataset (Bell et al., CVPR 2015)
- This project and any contributors
