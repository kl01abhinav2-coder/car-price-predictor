# Car Price Prediction (Flask ML App)
A production-ready Flask-based machine learning web application that predicts the market selling price of used cars based on user-provided features. The project combines a trained Random Forest Regression model with a modern, cinematic UI and is deployed live on Railway.
## Live Link
https://web-production-86706.up.railway.app/

##  Sample Demo



##  Features

- Machine Learning–powered price prediction  
- Clean & cinematic glassmorphism UI  
- Trained *Random Forest Regressor* model  
- Flask backend with *Jinja2* templating  
- Deployed on *Railway* (production-ready)  
- Input validation & safe prediction handlin

- ##  Machine Learning Model

- *Algorithm:* Random Forest Regressor  
- *Framework:* scikit-learn  
- *Target Variable:* Selling Price (in Lakhs)  
- *Model File:* rfr_model.pkl

- ##  Tech Stack

- *Backend:* Python 3, Flask, NumPy, scikit-learn  
- *Frontend:* HTML5, CSS3 (Glassmorphism UI), JavaScript  
- *Deployment:* Railway

- ## How to Run Locally
### 1. Clone the repository
```bash
git clone [https://github.com/Renju-rl/car_price.git](https://github.com/Renju-rl/car_price.git)
cd car_price

#### 2 Create virtual environment
bash
python -m venv venv
venv\Scripts\activate

#### 3 Install dependencies
bash
pip install -r requirements.txt

#### 4 Run the application
bash
python app.py

