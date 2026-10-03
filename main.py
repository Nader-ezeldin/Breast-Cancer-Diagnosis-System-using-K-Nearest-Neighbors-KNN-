import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class BreastCancerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Diagnosis System - KNN")
        self.root.geometry("950x800")
        self.root.configure(bg="#f0f0f0")
        
        # Model variables
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_trained = False
        
        # Create interface
        self.create_widgets()
        
        # Load default data
        self.load_default_data()
    
    def create_widgets(self):
        # Main title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="🏥 Breast Cancer Diagnosis System using KNN Algorithm",
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack(pady=15)
        
        # Control frame
        control_frame = tk.LabelFrame(
            self.root, 
            text="Control Panel",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            padx=10,
            pady=10
        )
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Control buttons
        btn_load = tk.Button(
            control_frame,
            text="📂 Load New Data",
            command=self.load_data,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5
        )
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_train = tk.Button(
            control_frame,
            text="🔄 Train Model",
            command=self.train_model,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5
        )
        btn_train.pack(side=tk.LEFT, padx=5)
        
        btn_clear = tk.Button(
            control_frame,
            text="🗑️ Clear Fields",
            command=self.clear_inputs,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=5
        )
        btn_clear.pack(side=tk.LEFT, padx=5)
        
        # Input frame
        input_frame = tk.LabelFrame(
            self.root,
            text="Enter Cell Measurements Values",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            padx=10,
            pady=10
        )
        input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Define features
        self.features = [
            ("radius_mean", "Mean Radius"),
            ("texture_mean", "Mean Texture"),
            ("perimeter_mean", "Mean Perimeter"),
            ("area_mean", "Mean Area"),
            ("smoothness_mean", "Mean Smoothness"),
            ("compactness_mean", "Mean Compactness"),
            ("concavity_mean", "Mean Concavity"),
            ("concave points_mean", "Mean Concave Points")
        ]
        
        self.entries = {}
        
        # Create input fields in grid
        for i, (key, label) in enumerate(self.features):
            row = i // 2
            col = (i % 2) * 2
            
            lbl = tk.Label(
                input_frame,
                text=f"{label} ({key}):",
                font=("Arial", 10),
                bg="#ecf0f1",
                anchor="w"
            )
            lbl.grid(row=row, column=col, padx=10, pady=5, sticky="w")
            
            entry = tk.Entry(
                input_frame,
                width=15,
                font=("Arial", 10),
                relief=tk.SOLID,
                borderwidth=1
            )
            entry.grid(row=row, column=col+1, padx=10, pady=5)
            self.entries[key] = entry
        
        # Predict button
        btn_predict = tk.Button(
            self.root,
            text="🔍 Diagnose Tumor",
            command=self.predict,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 14, "bold"),
            padx=30,
            pady=10
        )
        btn_predict.pack(pady=15)
        
        # Result frame
        result_frame = tk.LabelFrame(
            self.root,
            text="Diagnosis Result",
            font=("Arial", 12, "bold"),
            bg="#ecf0f1",
            padx=10,
            pady=10
        )
        result_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.result_label = tk.Label(
            result_frame,
            text="Please enter values and click Diagnose button",
            font=("Arial", 12),
            bg="#ecf0f1",
            fg="#34495e"
        )
        self.result_label.pack(pady=10)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Status: Model not trained",
            font=("Arial", 9),
            bg="#34495e",
            fg="white",
            anchor="w"
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)
    
    def load_default_data(self):
        """Load default data"""
        try:
            # Try to load file from same path
            if os.path.exists('knn_breast_cancer-selected-columns.csv'):
                self.df = pd.read_csv('knn_breast_cancer-selected-columns.csv')
                self.update_status(f"Data loaded: {len(self.df)} samples")
                self.train_model()
            else:
                messagebox.showwarning(
                    "Warning",
                    "Data file not found. Please load the file manually."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def load_data(self):
        """Load new CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.update_status(f"Loaded: {os.path.basename(file_path)} ({len(self.df)} samples)")
                messagebox.showinfo("Success", f"Successfully loaded {len(self.df)} samples!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def train_model(self):
        """Train KNN model"""
        try:
            if not hasattr(self, 'df'):
                messagebox.showwarning("Warning", "Please load data first")
                return
            
            # Update status bar
            self.update_status("Training model...")
            self.root.update()
            
            # Prepare data
            X = self.df.drop(['id', 'diagnosis'], axis=1)
            y = self.df['diagnosis']
            
            # Encode target variable
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            self.update_status(f"✓ Model trained successfully | Accuracy: {accuracy*100:.2f}%")
            
            messagebox.showinfo(
                "Success",
                f"Model trained successfully!\n\n"
                f"Model Accuracy: {accuracy*100:.2f}%\n"
                f"Training samples: {len(X_train)}\n"
                f"Testing samples: {len(X_test)}"
            )
            
        except Exception as e:
            self.update_status("✗ Model training failed")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def predict(self):
        """Predict tumor diagnosis"""
        if not self.is_trained:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        try:
            # Collect values from fields
            values = []
            for key, _ in self.features:
                value = self.entries[key].get().strip()
                if not value:
                    messagebox.showwarning(
                        "Warning",
                        f"Please enter a value for: {key}"
                    )
                    return
                
                try:
                    values.append(float(value))
                except ValueError:
                    messagebox.showerror(
                        "Error",
                        f"Invalid value for {key}. Please enter a number."
                    )
                    return
            
            # Convert to array
            X_new = np.array([values])
            
            # Scale features
            X_new_scaled = self.scaler.transform(X_new)
            
            # Predict
            prediction = self.model.predict(X_new_scaled)
            probabilities = self.model.predict_proba(X_new_scaled)
            
            # Get result
            result = self.label_encoder.inverse_transform(prediction)[0]
            confidence = np.max(probabilities) * 100
            
            # Display result
            if result == 'M':
                self.result_label.config(
                    text=f"⚠️ Diagnosis: Malignant Tumor\n"
                         f"Confidence: {confidence:.2f}%",
                    fg="#e74c3c",
                    font=("Arial", 14, "bold")
                )
            else:
                self.result_label.config(
                    text=f"✓ Diagnosis: Benign Tumor\n"
                         f"Confidence: {confidence:.2f}%",
                    fg="#27ae60",
                    font=("Arial", 14, "bold")
                )
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def clear_inputs(self):
        """Clear all input fields"""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        
        self.result_label.config(
            text="Please enter values and click Diagnose button",
            fg="#34495e",
            font=("Arial", 12)
        )
    
    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=f"Status: {message}")


def main():
    root = tk.Tk()
    app = BreastCancerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
