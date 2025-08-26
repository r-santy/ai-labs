# Prerequisites

- Azure ML Studio workspace access
- GitHub repository cloned to your local machine

# Setup Instructions
**Step 1: Launch Compute Instance**

1. Open Azure ML Studio (ml.azure.com)
2. Go to Compute → Compute instances
3. Click on your compute instance name
4. Click JupyterLab to launch

**Step 2: Create Project Folder**

1. In JupyterLab, click the Folder icon to create new folder
2. Name it: sarimax-lab
3. Double-click to enter the folder

**Step 3: Upload Files**

1. Click the Upload button (↑ icon) in JupyterLab
2. Upload these files from your local GitHub clone:

- sarimax-bf-scaling.py
- workload-synthetic-bf-patched.csv

**Step 4: Create Virtual Environment**

1. Open a new Terminal in JupyterLab
2. Run these commands:

```bash
# Terminal commands
cd sarimax-lab
python -m venv bf-env
source bf-env/bin/activate
```

**Step 5: Install Dependencies**
pip install pandas numpy matplotlib statsmodels

**Step 6: Run the Code**

Upload code and data files from your local folder and run

```bash
# Terminal commands
python sarima-bf-scaling.py
```
# Expected Output

Training metrics: RMSE and MAPE values
Forecast plot showing actual vs predicted Black Friday traffic
Predicted no. of VMs
