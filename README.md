# German Traffic Sign Recognition Benchmark (GTRSB)

The **GTRSB dataset** (German Traffic Sign Recognition Benchmark) was first introduced as a **single-image classification challenge** at the *International Joint Conference on Neural Networks (IJCNN) 2011*.

## 📌 Dataset Properties  
- **43 Categories** of traffic signs  
- **51,389 Images** in total  

The dataset can be found at the following link:  
🔗https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign


## 🔍 Usage in Adversarial Robustness Testing  
This dataset was used to evaluate **adversarial robustness** against **Fast Gradient Sign Method (FGSM) attacks**, in conjunction with:  
- **Basic neural network training**  
- **Adversarial training**  

## 📊 Results  

### ✅ **Benign Accuracy**  
- **Training Set:** 95%  
- **Validation Set:** 96%  
- **Test Set:** 91%  

### ⚠️ **Adversarial Attacks & Robustness Testing**  
- **Benign accuracy on adversarial batch:** **69%**  
- **Post-Adversarial Training:**  
  - **Adversarial Batch:** **89%**  
  - **Test Set:** **78%**  
