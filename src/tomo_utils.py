import numpy as np


def test_hermitian(cov_matrix):
    return np.allclose(cov_matrix, np.conjugate(np.swapaxes(cov_matrix, 2, 3)))

def test_positive_semidefinite(cov_matrix):
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            eigvals = np.linalg.eigvals(cov_matrix[i, j])
            if not np.all(eigvals >= -1e-10):  # Allow for small numerical errors
                return False
    return True

    

def test_diagonal_real_positive(cov_matrix):
    """  test if diagnoal real if clost 
    """
    diag = np.diagonal(cov_matrix, axis1=2, axis2=3)
    real_diag = np.real_if_close(diag)
    return np.all(np.isreal(real_diag)) and np.all(real_diag > 0)


def test_correlation_coefficients(cov_matrix):
    n = cov_matrix.shape[0]
    m = cov_matrix.shape[2]
    
    # Compute standard deviations
    std_dev = np.sqrt(np.diagonal(cov_matrix, axis1=2, axis2=3))
    
    for i in range(n):
        for j in range(m):
            for k in range(m):
                if j == k:
                    continue  # Diagonal elements are always 1, no need to check
                corr = cov_matrix[i, :, j, k] / (std_dev[i, :, j] * std_dev[i, :, k])
                if np.any(np.abs(corr) > 1 + 1e-10):
                    return False
    
    return True

def test_trace_property(cov_matrix):
    trace = np.trace(cov_matrix, axis1=2, axis2=3)
    diag_sum = np.sum(np.diagonal(cov_matrix, axis1=2, axis2=3), axis=2)
    return np.allclose(trace, diag_sum)





def test_covariance_matrix(cov_matrix):
    """
    Run all tests on the given covariance matrix.
    
    Args:
    cov_matrix (numpy.ndarray): The covariance matrix to test.
    
    Returns:
    dict: A dictionary containing the results of each test.
    """
    results = {}
    
    # Test for Hermitian property
    results['hermitian'] = test_hermitian(cov_matrix)
    
    # Test for positive semi-definite property
    results['positive_semidefinite'] = test_positive_semidefinite(cov_matrix)
    
    # Test for real and positive diagonal elements
    results['diagonal_real_positive'] = test_diagonal_real_positive(cov_matrix)
    
    # Test for valid correlation coefficients
    results['correlation_coefficients'] = test_correlation_coefficients(cov_matrix)
    
    # Test for trace property
    results['trace_property'] = test_trace_property(cov_matrix)
    
    return results

# Helper function to print results in a readable format
def print_test_results(results):
    print("Covariance Matrix Test Results:")
    for test, result in results.items():
        print(f"{test.replace('_', ' ').title()}: {'Passed' if result else 'Failed'}")
    print("All tests passed!" if all(results.values()) else "Some tests failed.")

# Example usage:
# cov_matrix = ... # Your covariance matrix here
# results = test_covariance_matrix(cov_matrix)
# print_test_results(results)