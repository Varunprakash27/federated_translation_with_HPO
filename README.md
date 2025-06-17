🧠 Federated Translation with Hyperparameter Optimization (HPO)

This project demonstrates a federated learning pipeline for machine translation, enhanced with hyperparameter optimization (HPO) using Optuna.

------------------------------------------------------------

🚀 Features:

- 🌍 Federated Learning using Flower to train models across multiple devices without sharing raw data  
- 🎯 Hyperparameter Optimization with Optuna for tuning learning rate, batch size, and number of epochs  
- 📚 Transformer-based translation using MarianMT from HuggingFace  
- 🧪 Logs and stores results from HPO trials for reproducibility and analysis  

------------------------------------------------------------

📂 Project Structure:

federated_translation_with_HPO/  
├── .gitignore              - Ignores folders like venv, pycache, etc.  
├── fl_client.py            - Client-side federated training logic  
├── fl_server.py            - Server to coordinate federated rounds  
├── hpo_optuna.py           - Optuna setup and search strategy  
├── translate.py            - Translation and inference script  
├── utils.py                - Utility functions  
├── requirements.txt        - Required Python packages  
└── data/                   - Training and testing data  

------------------------------------------------------------

⚙️ Installation:

1. Set up a Python virtual environment  
2. Install the dependencies using:  
   pip install -r requirements.txt

------------------------------------------------------------

▶️ How to Run:

1. Start the federated server  
   python fl_server.py


2. Open two separate terminals for clients:
   In Terminal 1:
   python fl_client.py

   In Terminal 2:
   python fl_client.py

   This simulates two clients participating in the federated learning process.

3. To perform hyperparameter optimization  
   python hpo_optuna.py

4. For translation inference  
   python translate.py

------------------------------------------------------------

🛠 Technologies Used:

- Python  
- PyTorch  
- HuggingFace Transformers  
- Flower (FLWR)  
- Optuna  

------------------------------------------------------------

🚫 Git Ignore Info:

These files and folders are excluded from version control:  
- venv/  
- __pycache__/  
- *.pt (model files)  
- hpo_results/

------------------------------------------------------------

🤝 Contributing:

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements or bug fixes.

------------------------------------------------------------

📄 License:

This project is open-source and available under the MIT License.
