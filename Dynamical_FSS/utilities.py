import numpy as np

def gather_data(sample_size, layer, Nh, T, N, taus, mode="VQAtrain",
                interaction="fully_connected", system_ = "quantum", observable="mag2"):
    
    last_rows = np.empty(len(taus))
    var_last_rows = np.empty(len(taus))
    
    for i, tau in enumerate(taus):
        filepath = (f"../output_files/{interaction}_{system_}_ising_dynamical_fss/"
                    f"{interaction}_sample{sample_size}_layer{layer}_Nh{Nh}_tau{tau}/"
                    f"{mode}_{observable}_temperature={T}_N={N}.npy")
        var_filepath = (f"../output_files/{interaction}_{system_}_ising_dynamical_fss/"
                        f"{interaction}_sample{sample_size}_layer{layer}_Nh{Nh}_tau{tau}/"
                        f"{mode}_var{observable}_temperature={T}_N={N}.npy")
        
        try:
            # Load the data and get the last row
            data = np.load(filepath)
            var = np.load(var_filepath)
            last_row = data[-1]  # Get the last row
            var_last_row = var[-1]
            last_rows[i] = last_row
            var_last_rows[i] = var_last_row
        except (FileNotFoundError, IOError) as e:
            print(f"Error loading file for Nh={Nh}: {e}")
            last_rows[i] = None
            var_last_rows[i] = None
    
    return last_rows, var_last_rows