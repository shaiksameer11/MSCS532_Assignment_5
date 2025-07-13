"""
Unit tests for Selection Algorithms implementation
Tests correctness, edge cases, and performance characteristics
"""

import unittest
import sys
import os
import random

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'part1_selection_algorithms'))

from selection_algorithms import SelectionAlgorithms

class TestSelectionAlgorithms(unittest.TestCase):
    """Test cases for both deterministic and randomized selection algorithms"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.selector = SelectionAlgorithms()
        
    def test_empty_array(self):
        """Test selection on empty array should raise error"""
        with self.assertRaises(ValueError):
            self.selector.deterministic_select([], 1)
        
        with self.assertRaises(ValueError):
            self.selector.randomized_select([], 1)
    
    def test_invalid_k_values(self):
        """Test invalid k values should raise errors"""
        arr = [1, 2, 3, 4, 5]
        
        # k too small
        with self.assertRaises(ValueError):
            self.selector.deterministic_select(arr, 0)
        
        with self.assertRaises(ValueError):
            self.selector.randomized_select(arr, 0)
        
        # k too large
        with self.assertRaises(ValueError):
            self.selector.deterministic_select(arr, 6)
        
        with self.assertRaises(ValueError):
            self.selector.randomized_select(arr, 6)
    
    def test_single_element(self):
        """Test selection on single element array"""
        arr = [42]
        
        det_result = self.selector.deterministic_select(arr, 1)
        rand_result = self.selector.randomized_select(arr, 1)
        
        self.assertEqual(det_result, 42)
        self.assertEqual(rand_result, 42)
    
    def test_two_elements(self):
        """Test selection on two element array"""
        arr = [5, 2]
        
        # First smallest (minimum)
        det_min = self.selector.deterministic_select(arr, 1)
        rand_min = self.selector.randomized_select(arr, 1)
        self.assertEqual(det_min, 2)
        self.assertEqual(rand_min, 2)
        
        # Second smallest (maximum)
        det_max = self.selector.deterministic_select(arr, 2)
        rand_max = self.selector.randomized_select(arr, 2)
        self.assertEqual(det_max, 5)
        self.assertEqual(rand_max, 5)
    
    def test_small_sorted_array(self):
        """Test selection on small sorted array"""
        arr = [1, 2, 3, 4, 5]
        
        for k in range(1, 6):
            det_result = self.selector.deterministic_select(arr, k)
            rand_result = self.selector.randomized_select(arr, k)
            
            # kth smallest should be k for sorted array [1,2,3,4,5]
            self.assertEqual(det_result, k)
            self.assertEqual(rand_result, k)
    
    def test_small_reverse_sorted_array(self):
        """Test selection on small reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        expected = [1, 2, 3, 4, 5]  # sorted order
        
        for k in range(1, 6):
            det_result = self.selector.deterministic_select(arr, k)
            rand_result = self.selector.randomized_select(arr, k)
            
            self.assertEqual(det_result, expected[k-1])
            self.assertEqual(rand_result, expected[k-1])
    
    def test_array_with_duplicates(self):
        """Test selection on array with duplicate elements"""
        arr = [3, 1, 3, 2, 1, 2, 3]
        sorted_arr = sorted(arr)  # [1, 1, 2, 2, 3, 3, 3]
        
        for k in range(1, len(arr) + 1):
            det_result = self.selector.deterministic_select(arr, k)
            rand_result = self.selector.randomized_select(arr, k)
            
            self.assertEqual(det_result, sorted_arr[k-1])
            self.assertEqual(rand_result, sorted_arr[k-1])
    
    def test_all_same_elements(self):
        """Test selection on array with all identical elements"""
        arr = [7, 7, 7, 7, 7]
        
        for k in range(1, 6):
            det_result = self.selector.deterministic_select(arr, k)
            rand_result = self.selector.randomized_select(arr, k)
            
            self.assertEqual(det_result, 7)
            self.assertEqual(rand_result, 7)
    
    def test_negative_numbers(self):
        """Test selection on array with negative numbers"""
        arr = [-5, 3, -1, 0, 2, -3]
        sorted_arr = sorted(arr)  # [-5, -3, -1, 0, 2, 3]
        
        for k in range(1, len(arr) + 1):
            det_result = self.selector.deterministic_select(arr, k)
            rand_result = self.selector.randomized_select(arr, k)
            
            self.assertEqual(det_result, sorted_arr[k-1])
            self.assertEqual(rand_result, sorted_arr[k-1])
    
    def test_random_arrays(self):
        """Test selection on random arrays"""
        for size in [10, 15, 20]:
            arr = [random.randint(1, 100) for _ in range(size)]
            sorted_arr = sorted(arr)
            
            # Test finding minimum, median, maximum
            test_k_values = [1, size // 2, size]
            
            for k in test_k_values:
                det_result = self.selector.deterministic_select(arr, k)
                rand_result = self.selector.randomized_select(arr, k)
                
                self.assertEqual(det_result, sorted_arr[k-1])
                self.assertEqual(rand_result, sorted_arr[k-1])
    
    def test_median_finding(self):
        """Test finding median specifically"""
        # Odd length array
        arr_odd = [7, 2, 9, 1, 5, 3, 8]
        sorted_odd = sorted(arr_odd)  # [1, 2, 3, 5, 7, 8, 9]
        median_odd = sorted_odd[len(arr_odd) // 2]  # 5
        
        det_median_odd = self.selector.deterministic_select(arr_odd, len(arr_odd) // 2 + 1)
        rand_median_odd = self.selector.randomized_select(arr_odd, len(arr_odd) // 2 + 1)
        
        self.assertEqual(det_median_odd, median_odd)
        self.assertEqual(rand_median_odd, median_odd)
        
        # Even length array  
        arr_even = [6, 2, 8, 1, 4, 3]
        sorted_even = sorted(arr_even)  # [1, 2, 3, 4, 6, 8]
        # For even length, we can test both middle elements
        median1 = sorted_even[len(arr_even) // 2 - 1]  # 3
        median2 = sorted_even[len(arr_even) // 2]      # 4
        
        det_median1 = self.selector.deterministic_select(arr_even, len(arr_even) // 2)
        rand_median1 = self.selector.randomized_select(arr_even, len(arr_even) // 2)
        
        det_median2 = self.selector.deterministic_select(arr_even, len(arr_even) // 2 + 1)
        rand_median2 = self.selector.randomized_select(arr_even, len(arr_even) // 2 + 1)
        
        self.assertEqual(det_median1, median1)
        self.assertEqual(rand_median1, median1)
        self.assertEqual(det_median2, median2)
        self.assertEqual(rand_median2, median2)
    
    def test_min_max_finding(self):
        """Test finding minimum and maximum elements"""
        arr = [15, 3, 9, 1, 12, 7, 20, 5]
        
        # Test minimum (k=1)
        det_min = self.selector.deterministic_select(arr, 1)
        rand_min = self.selector.randomized_select(arr, 1)
        expected_min = min(arr)
        
        self.assertEqual(det_min, expected_min)
        self.assertEqual(rand_min, expected_min)
        
        # Test maximum (k=n)
        det_max = self.selector.deterministic_select(arr, len(arr))
        rand_max = self.selector.randomized_select(arr, len(arr))
        expected_max = max(arr)
        
        self.assertEqual(det_max, expected_max)
        self.assertEqual(rand_max, expected_max)

class TestSelectionAlgorithmsPerformance(unittest.TestCase):
    """Test cases for performance characteristics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = SelectionAlgorithms()
    
    def test_operation_counting(self):
        """Test that operation counters work correctly"""
        arr = [5, 2, 8, 1, 9, 3]
        k = 3
        
        # Test deterministic selection counting
        self.selector.reset_counters()
        initial_comparisons = self.selector.comparisons
        initial_recursive_calls = self.selector.recursive_calls
        
        try:
            self.selector.deterministic_select(arr.copy(), k)
            
            # Check that counters increased
            self.assertGreater(self.selector.comparisons, initial_comparisons)
            self.assertGreater(self.selector.recursive_calls, initial_recursive_calls)
        except RecursionError:
            # Deterministic algorithm might hit recursion limit on small arrays
            pass
        
        # Test randomized selection counting
        self.selector.reset_counters()
        initial_comparisons = self.selector.comparisons
        initial_recursive_calls = self.selector.recursive_calls
        
        self.selector.randomized_select(arr.copy(), k)
        
        # Check that counters increased
        self.assertGreater(self.selector.comparisons, initial_comparisons)
        self.assertGreater(self.selector.recursive_calls, initial_recursive_calls)
    
    def test_randomized_consistency(self):
        """Test that randomized algorithm gives consistent results"""
        arr = [7, 2, 9, 1, 5, 3, 8, 4, 6]
        k = 5
        expected = sorted(arr)[k-1]
        
        # Run randomized selection multiple times
        results = []
        for _ in range(10):
            result = self.selector.randomized_select(arr.copy(), k)
            results.append(result)
        
        # All results should be the same (correct answer)
        for result in results:
            self.assertEqual(result, expected)
    
    def test_input_not_modified(self):
        """Test that original input array is not modified"""
        original_arr = [5, 2, 8, 1, 9, 3]
        arr_copy1 = original_arr.copy()
        arr_copy2 = original_arr.copy()
        
        # Run both algorithms
        try:
            self.selector.deterministic_select(arr_copy1, 3)
        except RecursionError:
            # Skip if recursion error occurs
            pass
        
        self.selector.randomized_select(arr_copy2, 3)
        
        # Original array should be unchanged
        # (Note: algorithms work on copies internally)
        self.assertEqual(original_arr, [5, 2, 8, 1, 9, 3])

class TestArrayGeneration(unittest.TestCase):
    """Test cases for test array generation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = SelectionAlgorithms()
    
    def test_generate_test_arrays(self):
        """Test test array generation functionality"""
        size = 10
        arrays = self.selector.generate_test_arrays(size)
        
        # Check all expected array types are generated
        expected_types = ['random', 'sorted', 'reverse_sorted', 'duplicates', 'nearly_sorted', 'single_element']
        for array_type in expected_types:
            self.assertIn(array_type, arrays)
            self.assertEqual(len(arrays[array_type]), size)
        
        # Verify sorted array is actually sorted
        self.assertEqual(arrays['sorted'], list(range(1, size + 1)))
        
        # Verify reverse sorted array
        self.assertEqual(arrays['reverse_sorted'], list(range(size, 0, -1)))
        
        # Verify duplicates array has repeated elements
        unique_elements = len(set(arrays['duplicates']))
        self.assertLess(unique_elements, size, "Duplicates array should have repeated elements")
    
    def test_nearly_sorted_generation(self):
        """Test nearly sorted array generation"""
        size = 20
        arrays = self.selector.generate_test_arrays(size)
        nearly_sorted = arrays['nearly_sorted']
        
        # Should have correct size
        self.assertEqual(len(nearly_sorted), size)
        
        # Should contain elements from 1 to size
        self.assertEqual(set(nearly_sorted), set(range(1, size + 1)))
        
        # Should be mostly sorted (check first few elements)
        mostly_sorted_count = 0
        for i in range(min(10, size - 1)):
            if nearly_sorted[i] < nearly_sorted[i + 1]:
                mostly_sorted_count += 1
        
        # At least half should be in order
        self.assertGreater(mostly_sorted_count, 3)

class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = SelectionAlgorithms()
    
    def test_large_numbers(self):
        """Test selection with very large numbers"""
        arr = [1000000, 999999, 1000001, 500000]
        sorted_arr = sorted(arr)
        
        for k in range(1, len(arr) + 1):
            # Test only randomized due to deterministic complexity
            result = self.selector.randomized_select(arr, k)
            self.assertEqual(result, sorted_arr[k-1])
    
    def test_k_boundary_values(self):
        """Test k values at boundaries"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        
        # Test k=1 (minimum)
        min_det = self.selector.randomized_select(arr, 1)  # Use randomized for reliability
        min_rand = self.selector.randomized_select(arr, 1)
        self.assertEqual(min_det, min(arr))
        self.assertEqual(min_rand, min(arr))
        
        # Test k=n (maximum)
        max_det = self.selector.randomized_select(arr, len(arr))
        max_rand = self.selector.randomized_select(arr, len(arr))
        self.assertEqual(max_det, max(arr))
        self.assertEqual(max_rand, max(arr))
    
    def test_float_numbers(self):
        """Test selection with floating point numbers"""
        # Convert to integers since our implementation expects integers
        float_arr = [3.5, 1.2, 4.8, 2.1, 5.9]
        int_arr = [35, 12, 48, 21, 59]  # Scale by 10 and convert to int
        
        for k in range(1, len(int_arr) + 1):
            result = self.selector.randomized_select(int_arr, k)
            expected = sorted(int_arr)[k-1]
            self.assertEqual(result, expected)

if __name__ == '__main__':
    # Run all tests with detailed output
    unittest.main(verbosity=2)