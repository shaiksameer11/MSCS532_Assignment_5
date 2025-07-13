import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

class SelectionAlgorithms:
    """
    Implementation of deterministic and randomized selection algorithms
    for finding kth smallest element in an array
    """
    
    def __init__(self):
        self.comparisons = 0
        self.recursive_calls = 0
        
    def reset_counters(self):
        """Reset performance counters for analysis"""
        self.comparisons = 0
        self.recursive_calls = 0
    
    def deterministic_select(self, arr: List[int], k: int) -> int:
        """
        Deterministic selection algorithm (Median of Medians)
        Finds kth smallest element in worst-case O(n) time
        
        Parameters:
            arr: Input array
            k: Find kth smallest element (1-indexed)
            
        Returns:
            kth smallest element
        """
        if not arr or k < 1 or k > len(arr):
            raise ValueError("Invalid input parameters")
            
        return self._deterministic_select_helper(arr.copy(), 0, len(arr) - 1, k - 1)
    
    def _deterministic_select_helper(self, arr: List[int], left: int, right: int, k: int) -> int:
        """
        Helper function for deterministic selection
        
        Parameters:
            arr: Array to search in
            left: Left boundary
            right: Right boundary 
            k: Target index (0-indexed)
            
        Returns:
            kth smallest element
        """
        self.recursive_calls += 1
        
        # Base case: small arrays
        if right - left + 1 <= 5:
            # Sort small array and return kth element
            sub_arr = sorted(arr[left:right + 1])
            return sub_arr[k - left]
        
        # Step 1: Divide array into groups of 5
        medians = []
        for i in range(left, right + 1, 5):
            group_end = min(i + 4, right)
            group = sorted(arr[i:group_end + 1])
            median_idx = len(group) // 2
            medians.append(group[median_idx])
        
        # Step 2: Find median of medians recursively
        median_of_medians = self._deterministic_select_helper(
            medians, 0, len(medians) - 1, len(medians) // 2
        )
        
        # Step 3: Partition around median of medians
        pivot_idx = self._partition_around_value(arr, left, right, median_of_medians)
        
        # Step 4: Recursive call on appropriate side
        if k == pivot_idx:
            return arr[pivot_idx]
        elif k < pivot_idx:
            return self._deterministic_select_helper(arr, left, pivot_idx - 1, k)
        else:
            return self._deterministic_select_helper(arr, pivot_idx + 1, right, k)
    
    def randomized_select(self, arr: List[int], k: int) -> int:
        """
        Randomized selection algorithm (Randomized Quickselect)
        Finds kth smallest element in expected O(n) time
        
        Parameters:
            arr: Input array
            k: Find kth smallest element (1-indexed)
            
        Returns:
            kth smallest element
        """
        if not arr or k < 1 or k > len(arr):
            raise ValueError("Invalid input parameters")
            
        return self._randomized_select_helper(arr.copy(), 0, len(arr) - 1, k - 1)
    
    def _randomized_select_helper(self, arr: List[int], left: int, right: int, k: int) -> int:
        """
        Helper function for randomized selection
        
        Parameters:
            arr: Array to search in
            left: Left boundary
            right: Right boundary
            k: Target index (0-indexed)
            
        Returns:
            kth smallest element
        """
        self.recursive_calls += 1
        
        if left == right:
            return arr[left]
        
        # Choose random pivot
        pivot_idx = random.randint(left, right)
        
        # Partition around random pivot
        pivot_idx = self._randomized_partition(arr, left, right, pivot_idx)
        
        # Recursive call on appropriate side
        if k == pivot_idx:
            return arr[pivot_idx]
        elif k < pivot_idx:
            return self._randomized_select_helper(arr, left, pivot_idx - 1, k)
        else:
            return self._randomized_select_helper(arr, pivot_idx + 1, right, k)
    
    def _partition_around_value(self, arr: List[int], left: int, right: int, pivot_value: int) -> int:
        """
        Partition array around given pivot value
        
        Parameters:
            arr: Array to partition
            left: Left boundary
            right: Right boundary
            pivot_value: Value to partition around
            
        Returns:
            Final position of pivot
        """
        # Find pivot value in array
        pivot_idx = left
        for i in range(left, right + 1):
            if arr[i] == pivot_value:
                pivot_idx = i
                break
        
        return self._randomized_partition(arr, left, right, pivot_idx)
    
    def _randomized_partition(self, arr: List[int], left: int, right: int, pivot_idx: int) -> int:
        """
        Partition array around pivot element
        
        Parameters:
            arr: Array to partition
            left: Left boundary
            right: Right boundary
            pivot_idx: Index of pivot element
            
        Returns:
            Final position of pivot
        """
        # Move pivot to end
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        pivot_value = arr[right]
        
        # Partition using Lomuto scheme
        i = left - 1
        
        for j in range(left, right):
            self.comparisons += 1
            if arr[j] <= pivot_value:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        # Place pivot in correct position
        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        return i + 1
    
    def generate_test_arrays(self, size: int) -> dict:
        """
        Generate different types of test arrays
        
        Parameters:
            size: Size of arrays to generate
            
        Returns:
            Dictionary with different array types
        """
        return {
            'random': [random.randint(1, 1000) for _ in range(size)],
            'sorted': list(range(1, size + 1)),
            'reverse_sorted': list(range(size, 0, -1)),
            'duplicates': [random.randint(1, 10) for _ in range(size)],
            'nearly_sorted': self._generate_nearly_sorted(size),
            'single_element': [42] if size == 1 else [42] * size
        }
    
    def _generate_nearly_sorted(self, size: int) -> List[int]:
        """Generate nearly sorted array with few random swaps"""
        arr = list(range(1, size + 1))
        # Perform random swaps on 5% of elements
        for _ in range(max(1, size // 20)):
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
    
    def benchmark_algorithms(self, sizes: List[int], num_trials: int = 5) -> dict:
        """
        Compare performance of both selection algorithms
        
        Parameters:
            sizes: List of array sizes to test
            num_trials: Number of trials per test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'sizes': sizes,
            'deterministic': {'random': [], 'sorted': [], 'reverse_sorted': [], 'duplicates': []},
            'randomized': {'random': [], 'sorted': [], 'reverse_sorted': [], 'duplicates': []},
            'comparisons_det': {'random': [], 'sorted': [], 'reverse_sorted': [], 'duplicates': []},
            'comparisons_rand': {'random': [], 'sorted': [], 'reverse_sorted': [], 'duplicates': []}
        }
        
        for size in sizes:
            print(f"Testing with array size {size}...")
            
            for array_type in ['random', 'sorted', 'reverse_sorted', 'duplicates']:
                det_times = []
                rand_times = []
                det_comps = []
                rand_comps = []
                
                for trial in range(num_trials):
                    test_arrays = self.generate_test_arrays(size)
                    arr = test_arrays[array_type]
                    k = size // 2  # Find median
                    
                    # Test Deterministic Selection
                    self.reset_counters()
                    start_time = time.perf_counter()
                    try:
                        self.deterministic_select(arr.copy(), k)
                        end_time = time.perf_counter()
                        det_times.append(end_time - start_time)
                        det_comps.append(self.comparisons)
                    except:
                        # Handle recursion limit or other errors
                        det_times.append(float('inf'))
                        det_comps.append(0)
                    
                    # Test Randomized Selection
                    self.reset_counters()
                    start_time = time.perf_counter()
                    self.randomized_select(arr.copy(), k)
                    end_time = time.perf_counter()
                    rand_times.append(end_time - start_time)
                    rand_comps.append(self.comparisons)
                
                # Store average results (excluding infinite values)
                valid_det_times = [t for t in det_times if t != float('inf')]
                if valid_det_times:
                    results['deterministic'][array_type].append(np.mean(valid_det_times))
                    results['comparisons_det'][array_type].append(np.mean(det_comps))
                else:
                    results['deterministic'][array_type].append(float('inf'))
                    results['comparisons_det'][array_type].append(0)
                
                results['randomized'][array_type].append(np.mean(rand_times))
                results['comparisons_rand'][array_type].append(np.mean(rand_comps))
        
        return results
    
    def plot_comparison_results(self, results: dict):
        """Create graphs showing algorithm comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        sizes = results['sizes']
        array_types = ['random', 'sorted', 'reverse_sorted', 'duplicates']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, array_type in enumerate(array_types):
            ax = [ax1, ax2, ax3, ax4][i]
            
            det_times = results['deterministic'][array_type]
            rand_times = results['randomized'][array_type]
            
            # Only plot finite values
            valid_det_indices = [j for j, t in enumerate(det_times) if t != float('inf')]
            valid_det_sizes = [sizes[j] for j in valid_det_indices]
            valid_det_times = [det_times[j] for j in valid_det_indices]
            
            if valid_det_times:
                ax.plot(valid_det_sizes, valid_det_times, 'o-', 
                       color=colors[i], label='Deterministic', linewidth=2)
            
            ax.plot(sizes, rand_times, 's--', 
                   color=colors[i], label='Randomized', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Array Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{array_type.replace("_", " ").title()} Arrays')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('selection_algorithms_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def theoretical_analysis_demo(self):
        """
        Demonstrate theoretical analysis of selection algorithms
        """
        print("=== SELECTION ALGORITHMS THEORETICAL ANALYSIS ===\n")
        
        print("1. Deterministic Selection (Median of Medians):")
        print("==============================================")
        print("Algorithm Steps:")
        print("1. Divide array into groups of 5 elements")
        print("2. Find median of each group")
        print("3. Recursively find median of medians")
        print("4. Partition around median of medians")
        print("5. Recursively search appropriate side")
        print()
        print("Why it works in O(n) worst-case time:")
        print("- Step 1-2: O(n) time to process all groups")
        print("- Step 3: T(n/5) time to find median of medians")
        print("- Step 4: O(n) time for partitioning")
        print("- Step 5: T(7n/10) time in worst case")
        print()
        print("Recurrence relation: T(n) = T(n/5) + T(7n/10) + O(n)")
        print("Since 1/5 + 7/10 = 19/20 < 1, this gives T(n) = O(n)")
        print()
        
        print("2. Randomized Selection (Randomized Quickselect):")
        print("=================================================")
        print("Algorithm Steps:")
        print("1. Choose random pivot element")
        print("2. Partition array around pivot")
        print("3. Recursively search appropriate side")
        print()
        print("Why it works in O(n) expected time:")
        print("- Random pivot gives good split with high probability")
        print("- Expected recurrence: T(n) = T(3n/4) + O(n)")
        print("- This gives expected time T(n) = O(n)")
        print()
        print("Expected vs Worst-case:")
        print("- Expected case: O(n) with high probability")
        print("- Worst case: O(n^2) if always pick bad pivots")
        print("- But probability of worst case is very low")
        print()
        
        # Empirical validation
        print("Empirical Validation:")
        print("====================")
        sizes = [100, 500, 1000, 2000]
        
        for size in sizes:
            arr = [random.randint(1, 1000) for _ in range(size)]
            k = size // 2
            
            # Test randomized selection multiple times
            total_comps = 0
            trials = 10
            
            for _ in range(trials):
                self.reset_counters()
                self.randomized_select(arr.copy(), k)
                total_comps += self.comparisons
            
            avg_comps = total_comps / trials
            linear_prediction = size * 2  # Rough linear estimate
            
            print(f"Array size {size}: Average comparisons = {avg_comps:.0f}, "
                  f"Linear estimate = {linear_prediction}, "
                  f"Ratio = {avg_comps/linear_prediction:.2f}")


# Example usage and testing
if __name__ == "__main__":
    selector = SelectionAlgorithms()
    
    # Basic demonstration
    print("=== BASIC SELECTION ALGORITHMS DEMONSTRATION ===")
    test_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    k = 5
    
    print(f"Original array: {test_array}")
    print(f"Finding {k}th smallest element...")
    
    det_result = selector.deterministic_select(test_array.copy(), k)
    rand_result = selector.randomized_select(test_array.copy(), k)
    
    print(f"Deterministic result: {det_result}")
    print(f"Randomized result: {rand_result}")
    print(f"Verification (sorted): {sorted(test_array)[k-1]}")
    print()
    
    # Theoretical analysis
    selector.theoretical_analysis_demo()
    
    print("\n" + "="*60)
    print("ALGORITHM PERFORMANCE COMPARISON")
    print("="*60)
    
    # Performance comparison
    sizes = [50, 100, 200, 500]  # Smaller sizes due to deterministic algorithm complexity
    results = selector.benchmark_algorithms(sizes, num_trials=3)
    
    # Create comparison graphs
    selector.plot_comparison_results(results)
    
    # Print summary
    print("\nPerformance Summary:")
    print("===================")
    for array_type in ['random', 'sorted', 'reverse_sorted', 'duplicates']:
        print(f"\n{array_type.replace('_', ' ').title()} Arrays:")
        det_avg = np.mean([t for t in results['deterministic'][array_type] if t != float('inf')])
        rand_avg = np.mean(results['randomized'][array_type])
        
        if det_avg != float('inf'):
            print(f"  Deterministic average: {det_avg:.6f} seconds")
        else:
            print(f"  Deterministic: Too slow (timeout)")
        print(f"  Randomized average: {rand_avg:.6f} seconds")