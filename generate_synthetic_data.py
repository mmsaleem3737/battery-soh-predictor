import pandas as pd
import numpy as np

def generate_single_cycle_data(cycle_number, degradation_factor=0.0):
    """
    Generates a single, realistic charge/discharge cycle.
    
    Args:
        cycle_number (int): The index of the cycle to generate.
        degradation_factor (float): A factor from 0.0 to 1.0 to simulate aging.
    """
    # Parameters for the cycle simulation
    charge_current = 0.5
    discharge_current = -0.5 * (1 - degradation_factor * 0.1) # Discharge current weakens with age
    min_voltage = 3.0
    max_voltage = 4.2 * (1 - degradation_factor * 0.05) # Max voltage drops with age
    
    n_points_charge = 100
    n_points_discharge = 100
    
    # --- Create Phases ---
    # 1. Charging Phase
    charge_voltage = np.linspace(min_voltage, max_voltage, n_points_charge)
    charge_current_arr = np.full(n_points_charge, charge_current)
    
    # 2. Discharging Phase
    discharge_voltage = np.linspace(max_voltage, min_voltage, n_points_discharge)
    discharge_current_arr = np.full(n_points_discharge, discharge_current)
    
    # Combine phases
    voltage = np.concatenate([charge_voltage, discharge_voltage])
    current = np.concatenate([charge_current_arr, discharge_current_arr])
    
    # Create the DataFrame for this cycle
    cycle_df = pd.DataFrame({
        'Cycle_Index': cycle_number,
        'Voltage (V)': voltage,
        'Current (A)': current,
    })
    
    # Add some noise to make it more realistic
    cycle_df['Voltage (V)'] += np.random.normal(0, 0.01, size=len(cycle_df))
    cycle_df['Current (A)'] += np.random.normal(0, 0.005, size=len(cycle_df))
    
    return cycle_df

def create_synthetic_dataset(num_cycles=5):
    """
    Creates a full dataset with multiple, progressively degrading cycles.
    """
    all_cycles = []
    for i in range(1, num_cycles + 1):
        # Apply more degradation to later cycles
        degradation = (i-1) / num_cycles
        cycle_data = generate_single_cycle_data(i, degradation_factor=degradation)
        all_cycles.append(cycle_data)
        
    # Combine all cycles into one DataFrame
    final_df = pd.concat(all_cycles, ignore_index=True)
    
    # Save to CSV
    file_path = 'synthetic_battery_data.csv'
    final_df.to_csv(file_path, index=False)
    
    print(f"Successfully generated synthetic data with {num_cycles} cycles.")
    print(f"File saved as: {file_path}")

if __name__ == "__main__":
    create_synthetic_dataset()
