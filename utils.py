from tensorflow import dtypes


def check_complex_dtype(dtype):
    if dtypes.as_dtype(dtype) not in [dtypes.complex64, dtypes.complex128]:
        raise ValueError("Dtype must be complex64 or complex128.")
