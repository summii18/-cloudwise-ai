# CloudWise AI

## 1. Project Overview

CloudWise AI is a machine-learning-based cloud resource recommendation system. It analyzes resource usage metrics and provides recommendations such as **Keep Running, Sleep Mode, and Shutdown**.

The project demonstrates how machine learning and cloud-related technologies can support resource monitoring and cost optimization.

## 2. Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas and NumPy
- FastAPI
- Uvicorn
- Requests
- Psutil
- Matplotlib

## 3. Main Features

- **Resource Advisor:** Predicts resource recommendations using a trained Decision Tree model.
- **Simulation Mode:** Simulates multiple virtual servers and displays changing resource metrics.
- **Model Performance:** Displays model accuracy, confusion matrix, and classification report.
- **Dataset Insights:** Displays dataset information, statistics, and visualizations.
- **Cloud Cost Estimator:** Estimates possible savings using user-provided cost and percentage values.
- **Connected Device Monitoring:** Collects CPU, RAM, disk, and process information from a Windows device through a monitoring agent and FastAPI backend.

## 4. Project Architecture

1. The CSV dataset provides resource usage information.
2. A Decision Tree model learns patterns from the dataset.
3. Streamlit provides the user interface.
4. FastAPI receives monitoring information from the connected device.
5. The monitoring agent collects device metrics using Psutil.
6. The dashboard displays predictions, analytics, simulations, and monitoring information.

## 5. Dataset and Model

**Dataset:** `data/Cost_Optimization.csv`

**Model:** Decision Tree Classifier

**Model parameters:**
- Maximum depth: 6
- Random state: 42
- Input features: CPU usage, RAM usage, active jobs, and idle time.

The model achieved **99.88% accuracy** on a held-out test set of 4,000 samples.

This result measures performance on the selected dataset split and does not guarantee the same accuracy on real-world cloud workloads.

## 6. Installation and Setup

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

## 7. Running the Project

The project uses three terminals.

### Terminal 1: Start the FastAPI backend

```powershell
uvicorn backend:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

### Terminal 2: Start the monitoring agent

```powershell
python agent.py
```

### Terminal 3: Start the Streamlit application

```powershell
streamlit run app.py
```

Open the application:

`http://localhost:8501`

## 8. Important Project Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `backend.py` | FastAPI backend |
| `agent.py` | Device monitoring agent |
| `simulation.py` | Virtual server simulation |
| `cloudwise_model.pkl` | Saved machine learning model |
| `requirements.txt` | Project dependencies |
| `data/Cost_Optimization.csv` | Dataset |

## 9. Limitations

- The project is an educational prototype.
- The cost estimator uses user-provided assumptions and is not connected to AWS billing.
- Connected laptop metrics are displayed separately and are not directly used by the trained model because their feature definitions differ from the training dataset.
- The system does not automatically shut down or modify any real cloud server.
- Model performance depends on the quality and characteristics of the dataset.

## 10. Future Enhancements

- Integration with AWS EC2 monitoring.
- Cloud billing data integration.
- Improved model training with real-world cloud datasets.
- Advanced resource usage forecasting.
- Secure cloud deployment.