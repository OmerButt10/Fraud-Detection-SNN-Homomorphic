import time
import base64
import pandas as pd # type: ignore
import tenseal as ts  # type: ignore
from concrete.ml.torch.compile import compile_torch_model  # Ensure Concrete ML is installed  # type: ignore


# -------------------------
# Helper functions for FHE processing
# -------------------------

def get_quantize_module(model_inf, torch_input, method="exact", bits=8):
    start = time.time()
    return compile_torch_model(
        model_inf,
        torch_input,
        n_bits=bits,
        rounding_threshold_bits={"n_bits": 6, "method": method}
    ), round(time.time() - start, 2)

def get_quantize_input(quantized_module, X_test_spikes):
    start = time.time()
    return quantized_module.quantize_input(X_test_spikes.detach().cpu().numpy()), round(time.time() - start, 2)


def get_encrypted_input(quantized_module, q_input):
    # Encrypt the input
    start = time.time()
    return quantized_module.fhe_circuit.encrypt(q_input[0:1, :, :]), round(time.time() - start, 2)

def get_encrypted_output(quantized_module, q_input_enc):
    # Execute the linear product in FHE
    start = time.time()
    return quantized_module.fhe_circuit.run(q_input_enc), round(time.time() - start, 2)

def get_decrypted_output(quantized_module, q_y_enc):
    # Decrypt the result (integer)
    start = time.time()
    return quantized_module.fhe_circuit.decrypt(q_y_enc), round(time.time() - start, 2)

def get_dequantize_output(quantized_module, q_y):
    # De-quantize and post-process the result
    start = time.time()
    return quantized_module.post_processing(quantized_module.dequantize_output(q_y)), round(time.time() - start, 2)

def get_normal_output(model_inf, X_test_spikes):
    start = time.time()
    return model_inf.forward(X_test_spikes), round(time.time() - start, 2)


def create_tenseal_context():
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,  # Use CKKS for real numbers
        poly_modulus_degree=8192,  # Increase for better security but more overhead
        coeff_mod_bit_sizes=[60, 40, 40, 60],  # Security parameters
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context


def encrypt_with_tenseal(input_tensor, context):
    # Encrypt the input tensor using TenSEAL
    encrypted_tensor = ts.ckks_vector(context, input_tensor.flatten().tolist())
    return encrypted_tensor

def compare_encrypted_inputs(tenseal_enc_input, q_input_enc, input_buffer, encrypted_input_str):
    """
    Compares the encrypted input values from TenSEAL and FHE pipelines and writes the results to a CSV file.

    Parameters:
      - tenseal_enc_input: The TenSEAL encrypted input object.
      - q_input_enc: The FHE encrypted input object.
      - input_buffer: The serialized bytes from the FHE encrypted input.
      - encrypted_input_str: The base64-encoded string from the FHE encrypted input.

    The function creates a CSV file with:
      - A textual representation of the objects.
      - Their serialized bytes.
      - Their base64 encoded versions.
    """
    # TenSEAL values
    tenseal_obj_str = str(tenseal_enc_input)
    tenseal_serialized = tenseal_enc_input.serialize()  # bytes
    tenseal_base64 = base64.b64encode(tenseal_serialized).decode("utf-8")
    
    # FHE values (assumed to be already computed)
    fhe_obj_str = str(q_input_enc)
    fhe_serialized = input_buffer  # bytes, already computed
    fhe_base64 = encrypted_input_str  # already base64 encoded string

    # Print details for debugging and length checking.
    print("\n--- Encrypted Input Comparison ---")
    print("TenSEAL Object:", tenseal_obj_str)
    print("FHE Object:", fhe_obj_str)
    print("TenSEAL Serialized Bytes:", tenseal_serialized)
    print("FHE Serialized Bytes:", fhe_serialized)
    print("TenSEAL Serialized Bytes Length:", len(tenseal_serialized))
    print("FHE Serialized Bytes Length:", len(fhe_serialized))
    print("TenSEAL Base64 Length:", len(tenseal_base64))
    print("FHE Base64 Length:", len(fhe_base64))
    
    # Create a dictionary with the comparison details.
    data = {
        "Comparison": ["Object", "Serialized Bytes", "Base64"],
        "TenSEAL": [tenseal_obj_str, str(tenseal_serialized), tenseal_base64],
        "FHE": [fhe_obj_str, str(fhe_serialized), fhe_base64],
    }

    # Convert to DataFrame and write to CSV.
    df = pd.DataFrame(data)
    df.to_csv("encrypted_input_comparison.csv", index=False)
    print("CSV file 'encrypted_input_comparison.csv' created successfully.")


def compare_encrypted_outputs(tenseal_enc_output, q_y_enc, output_buffer, encrypted_output_str):
    """
    Compares the encrypted output values from TenSEAL and FHE pipelines and writes the results to a CSV file.

    Parameters:
      - tenseal_enc_output: The TenSEAL encrypted output object.
      - q_y_enc: The FHE encrypted output object.
      - output_buffer: The serialized bytes from the FHE encrypted output.
      - encrypted_output_str: The base64-encoded string from the FHE encrypted output.

    The function creates a CSV file with:
      - A textual representation of the objects.
      - Their serialized bytes.
      - Their base64 encoded versions.
    """
    # TenSEAL values
    tenseal_obj_str = str(tenseal_enc_output)
    tenseal_serialized = tenseal_enc_output.serialize()  # bytes from TenSEAL
    tenseal_base64 = base64.b64encode(tenseal_serialized).decode("utf-8")
    
    # FHE values (assumed to be already computed)
    fhe_obj_str = str(q_y_enc)
    fhe_serialized = output_buffer  # bytes, already computed
    fhe_base64 = encrypted_output_str  # already base64 encoded string

    # Print details for debugging and length checking.
    print("\n--- Encrypted Output Comparison ---")
    print("TenSEAL Object (Output):", tenseal_obj_str)
    print("FHE Object (Output):", fhe_obj_str)
    print("TenSEAL Serialized Bytes (Output):", tenseal_serialized)
    print("FHE Serialized Bytes (Output):", fhe_serialized)
    print("TenSEAL Serialized Bytes Length (Output):", len(tenseal_serialized))
    print("FHE Serialized Bytes Length (Output):", len(fhe_serialized))
    print("TenSEAL Base64 Length (Output):", len(tenseal_base64))
    print("FHE Base64 Length (Output):", len(fhe_base64))
    
    # Create a dictionary with the comparison details.
    data = {
        "Comparison": ["Object", "Serialized Bytes", "Base64"],
        "TenSEAL": [tenseal_obj_str, str(tenseal_serialized), tenseal_base64],
        "FHE": [fhe_obj_str, str(fhe_serialized), fhe_base64],
    }
    
    # Convert to DataFrame and write to CSV.
    df = pd.DataFrame(data)
    df.to_csv("encrypted_output_comparison.csv", index=False)
    print("CSV file 'encrypted_output_comparison.csv' created successfully.")
