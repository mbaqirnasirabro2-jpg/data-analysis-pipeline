"""
Title: Exploratory Data Analysis (EDA) and Automated Data Cleaning Pipeline
Author: Muhammad Baqir (GitHub: @mbaqirnasirabro2-jpg)
Description: A robust Python data analysis script built using Pandas and NumPy 
             to handle missing values, detect outliers, perform statistical aggregations, 
             and extract actionable insights from raw tabular data.
"""

import pandas as pd
import numpy as np

class DataAnalysisPipeline:
    def __init__(self, file_path=None):
        """Initializes the data pipeline."""
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads dataset from CSV or generates a sample dataset if path is not provided."""
        if self.file_path:
            try:
                self.df = pd.read_csv(self.file_path)
                print(f"[SUCCESS] Dataset loaded successfully from {self.file_path}")
            except Exception as e:
                print(f"[ERROR] Could not load file: {e}")
        else:
            print("[INFO] No file path provided. Generating sample dataset for demonstration...")
            # Generating a synthetic dataset for pipeline demonstration
            np.random.seed(42)
            n_rows = 1000
            self.df = pd.DataFrame({
                'transaction_id': range(101, 101 + n_rows),
                'category': np.random.choice(['Electronics', 'Solar Equipment', 'Apparel', 'Home Appliances'], n_rows),
                'unit_price': np.random.uniform(50.0, 1500.0, n_rows),
                'quantity': np.random.choice([1, 2, 3, 4, 5], n_rows),
                'customer_rating': np.random.choice([1, 2, 3, 4, 5, np.nan], n_rows), # Contains missing values
                'order_date': pd.date_range(start='2025-01-01', periods=n_rows, freq='h')
            })
            # Introduce some intentional missing values & duplicates for cleaning demo
            self.df.loc.iloc[10:20, 2] = np.nan 
            print("[SUCCESS] Sample dataset generated successfully!\n")

    def inspect_data(self):
        """Performs initial inspection of data structures, types, and missing values."""
        print("--- 1. Data Inspection & Summary ---")
        print(f"Dataset Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")
        print("Data Types & Non-Null Counts:")
        print(self.df.info())
        print("\nMissing Values Count per Column:")
        print(self.df.isnull().sum())
        print("-" * 40 + "\n")

    def clean_data(self):
        """Cleans data by handling missing values, fixing data types, and removing duplicates."""
        print("--- 2. Data Cleaning & Preprocessing ---")
        initial_rows = len(self.df)
        
        # 1. Handle missing numeric values using median imputation
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"Imputed missing values in '{col}' with median: {median_val}")

        # 2. Drop duplicate rows if any
        self.df.drop_duplicates(inplace=True)
        dropped_rows = initial_rows - len(self.df)
        print(f"Removed duplicate rows: {dropped_rows}")
        print("Data cleaning completed successfully!\n" + "-" * 40 + "\n")

    def perform_eda_aggregations(self):
        """Performs advanced aggregations and grouping to extract business insights."""
        print("--- 3. Exploratory Data Analysis & Aggregations ---")
        
        # Feature Engineering: Total Sale Value per transaction
        self.df['total_sales'] = self.df['unit_price'] * self.df['quantity']
        
        # Groupby Analysis: Sales by Category
        category_summary = self.df.groupby('category').agg(
            total_revenue=('total_sales', 'sum'),
            average_order_value=('total_sales', 'mean'),
            total_units_sold=('quantity', 'sum'),
            transaction_count=('transaction_id', 'count')
        ).reset_index().sort_values(by='total_revenue', ascending=False)
        
        print("Summary Performance by Category:")
        print(category_summary.to_string(index=False))
        print("\n" + "-" * 40 + "\n")
        
        return category_summary

    def export_cleaned_data(self, output_filename="cleaned_dataset.csv"):
        """Exports the cleaned dataset to a CSV file."""
        self.df.to_csv(output_filename, index=False)
        print(f"[SUCCESS] Cleaned dataset successfully exported to '{output_filename}'\n")


# --- Execution Example ---
if __name__ == "__main__":
    print("=== STARTING DATA ANALYSIS PIPELINE ===\n")
    
    # Initialize pipeline
    pipeline = DataAnalysisPipeline() # Pass CSV path here if you have a real file e.g., 'data.csv'
    
    # Run Pipeline Steps
    pipeline.load_data()
    pipeline.inspect_data()
    pipeline.clean_data()
    summary_results = pipeline.perform_eda_aggregations()
    pipeline.export_cleaned_data("processed_sales_data.csv")
