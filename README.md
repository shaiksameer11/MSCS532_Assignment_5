# Selection Algorithms and Elementary Data Structures

A complete implementation and analysis of selection algorithms for finding kth smallest elements and fundamental data structures including arrays, stacks, queues, linked lists, and trees. This project demonstrates both theoretical analysis and practical performance of essential computer science concepts.

## Project Overview

This repository contains implementations and analysis of:

### **Part 1: Selection Algorithms**
- **Deterministic Selection (Median of Medians):** Guaranteed O(n) worst-case time
- **Randomized Selection (Randomized Quickselect):** Expected O(n) time with simpler implementation
- Mathematical analysis of both algorithms
- Performance comparison across different input types

### **Part 2: Elementary Data Structures**
- **Dynamic Arrays:** Resizable arrays with amortized O(1) operations
- **Matrices:** 2D array implementation with basic operations
- **Stacks:** LIFO data structure using arrays
- **Queues:** FIFO data structure using circular arrays
- **Linked Lists:** Dynamic node-based data structure
- **Rooted Trees:** Hierarchical data structure with traversals

## Repository Structure

```
MSCS532_ASSIGNMENT_5/
├── README.md                              # This file - complete project documentation
├── requirements.txt                       # Python dependencies
├── combined_analysis.py                    # Execute all implementations and analysis
│
├── selection_algorithms/                  # Selection Algorithms Implementation
│   ├── __init__.py
│   ├── selection_algorithms.py            # Main selection algorithms implementation
│   ├── README.md                          # Selection algorithms documentation
│   └── results/                           # Generated analysis results
│       └── selection_algorithms_comparison.png
│
├── data_structures/                       # Elementary Data Structures Implementation
│   ├── __init__.py
│   ├── elementary_structures.py           # All data structures implementation
│   ├── README.md                          # Data structures documentation
│   └── results/                           # Generated performance analysis
│       └── data_structures_performance.png
│
├── docs/                                  # Complete documentation
│   ├── selection_analysis.md              # Detailed selection algorithms analysis
│   ├── data_structures_analysis.md        # Detailed data structures analysis
│   └── combined_analysis_report.md        # Combined project analysis report
│
├── tests/                                 # Unit tests for all implementations
│   ├── __init__.py
│   ├── test_selection_algorithms.py       # Selection algorithms tests
│   └── test_data_structures.py            # Data structures tests
```

## Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- Required packages (install using requirements.txt)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shaiksameer11/MSCS532_Assignment_5.git
   cd MSCS532_Assignment_5
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Individual Components

#### Option 1: Run Selection Algorithms Analysis

```bash
cd selection_algorithms
python selection_algorithms.py
```

**What this does:**
- Demonstrates both deterministic and randomized selection
- Shows mathematical analysis with step-by-step explanations
- Compares performance on different input types
- Generates performance comparison graphs
- Validates theoretical predictions with empirical results

**Output files:**
- `selection_algorithms/results/selection_algorithms_comparison.png`
- Console output with detailed analysis and statistics

#### Option 2: Run Data Structures Analysis

```bash
cd data_structures
python elementary_structures.py
```

**What this does:**
- Demonstrates all implemented data structures
- Shows usage examples for arrays, stacks, queues, linked lists, trees
- Analyzes time complexity of operations
- Compares performance across different structures
- Generates performance analysis graphs

**Output files:**
- `data_structures/results/data_structures_performance.png`
- Console output with demonstrations and complexity analysis

#### Option 3: Run Complete Analysis

```bash
python combined_analysis.py
```

**What this does:**
- Executes both selection algorithms and data structures analysis
- Generates combined performance report
- Creates comprehensive analysis documentation
- Produces complete project results

**Output files:**
- All individual component results
- `docs/combined_analysis_report.md` - Combined analysis report

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific component tests
python -m pytest tests/test_selection_algorithms.py
python -m pytest tests/test_data_structures.py
```

## Key Results Summary

### Selection Algorithms Performance

| Algorithm | Worst Case | Expected Case | Space | Implementation |
|-----------|------------|---------------|-------|----------------|
| Deterministic | O(n) | O(n) | O(log n) | Complex |
| Randomized | O(n²) | O(n) | O(log n) | Simple |

**Key Finding:** Randomized selection is 3-5x faster in practice despite worse worst-case complexity.

### Data Structures Performance

| Operation | Array | Linked List | Stack | Queue | Tree |
|-----------|-------|-------------|-------|-------|------|
| Access | O(1) | O(n) | O(1) top | O(1) front | O(h) |
| Insert | O(n) middle | O(1) begin | O(1) | O(1) | O(h) |
| Delete | O(n) middle | O(1) begin | O(1) | O(1) | O(h) |
| Search | O(n) | O(n) | - | - | O(n) |

**Key Finding:** Each data structure excels in specific operations - choose based on access patterns.

## Mathematical Analysis Highlights

### Selection Algorithms Complexity

**Deterministic Selection (Median of Medians):**
```
T(n) = T(n/5) + T(7n/10) + O(n)
```
Since 1/5 + 7/10 < 1, this gives T(n) = O(n) worst-case.

**Randomized Selection:**
```
E[T(n)] = E[T(3n/4)] + O(n)
```
Expected linear time due to good pivot probability ≥ 1/2.

### Data Structures Space-Time Trade-offs

**Arrays:** Fast access O(1) but expensive middle operations O(n)
**Linked Lists:** Fast begin operations O(1) but slow access O(n)  
**Stacks/Queues:** Specialized access patterns with O(1) operations
**Trees:** Hierarchical organization with O(h) operations

## Implementation Features

### Selection Algorithms Features

- **Complete median-of-medians implementation** with proper partitioning
- **Randomized quickselect** with random pivot selection
- **Comprehensive testing** on various input distributions
- **Performance monitoring** with operation counting
- **Statistical validation** through multiple trials

### Data Structures Features

- **Dynamic arrays** with automatic resizing and amortized analysis
- **Circular queue** implementation preventing unnecessary shifting
- **Generic linked list** supporting arbitrary data types
- **Tree implementation** with multiple traversal methods
- **Performance benchmarking** across all structures

## Testing Methodology

### Comprehensive Test Coverage

**Selection Algorithms Testing:**
- Array sizes: 50, 100, 200, 500 elements
- Input types: Random, sorted, reverse-sorted, duplicates
- Multiple k values for comprehensive validation
- Statistical averaging over multiple trials

**Data Structures Testing:**
- Performance scaling analysis across different sizes
- Operation-specific benchmarking (insert, delete, access)
- Memory usage analysis and space complexity validation
- Real-world usage pattern simulation

### Statistical Methodology

- **Multiple trials** for statistical reliability (5-10 runs per test)
- **Performance averaging** with standard deviation calculation
- **Complexity validation** through scaling analysis
- **Empirical vs theoretical** comparison

## Customization and Extension

### Modifying Test Parameters

**Selection Algorithms:**
```python
# Adjust test sizes and parameters
sizes = [50, 100, 200, 500, 1000]
num_trials = 10

# Test different k values
k_values = [1, size//4, size//2, 3*size//4, size]
```

**Data Structures:**
```python
# Modify benchmark sizes
sizes = [100, 500, 1000, 2000, 5000, 10000]

# Adjust operation counts for testing
operation_counts = [100, 500, 1000]
```

### Adding New Features

**Potential Extensions:**
1. **Advanced Selection:** Implement parallel selection algorithms
2. **Enhanced Data Structures:** Add doubly linked lists, deques
3. **Tree Variants:** Implement binary search trees, balanced trees
4. **Performance Optimization:** Add memory pool allocation
5. **Visualization:** Interactive data structure animations

## Outcomes

### Computer Science Concepts Demonstrated

1. **Algorithm Analysis:** Worst-case vs expected-case complexity
2. **Randomization:** How randomness improves average performance
3. **Data Structure Design:** Trade-offs between time and space
4. **Amortized Analysis:** Understanding average cost over operations
5. **Empirical Validation:** Testing theoretical predictions

### Learning Outcomes

- Understanding of fundamental selection algorithms and their analysis
- Practical experience with essential data structures
- Knowledge of performance trade-offs in algorithm and structure choice
- Skills in implementing and analyzing computer science fundamentals
- Appreciation for the connection between theory and practice

## Common Use Cases

### Selection Algorithms Applications

- **Statistics:** Finding medians and percentiles in large datasets
- **Database Systems:** Top-k queries and ranking operations
- **Graphics:** Color quantization and image processing
- **Machine Learning:** Feature selection and outlier detection

### Data Structures Applications

- **Arrays:** Scientific computing, image processing, game boards
- **Linked Lists:** Music playlists, undo functionality, memory management
- **Stacks:** Expression parsing, function calls, browser history
- **Queues:** Process scheduling, network buffering, event simulation
- **Trees:** File systems, decision trees, organizational hierarchies

## Further Reading

### Online Resources

1. **Programiz - Selection Algorithms**
   - https://www.programiz.com/dsa/quick-select
   - Clear explanation of quickselect with examples

2. **TutorialsPoint - Data Structures**
   - https://www.tutorialspoint.com/data_structures_algorithms/
   - Comprehensive tutorials on all fundamental structures

3. **InterviewBit - Algorithms and Data Structures**
   - https://www.interviewbit.com/tutorial/
   - Interview-focused content with practical examples

4. **Algorithm Visualizer**
   - https://algorithm-visualizer.org/
   - Interactive demonstrations of algorithms and structures

5. **Visualgo - Data Structures**
   - https://visualgo.net/en
   - Step-by-step animations of operations

### Technical Resources

6. **Python Documentation**
   - https://docs.python.org/3/tutorial/datastructures.html
   - Official guide to Python's built-in structures

7. **LeetCode Practice Problems**
   - https://leetcode.com/tag/array/
   - https://leetcode.com/tag/linked-list/
   - Hands-on coding practice

8. **HackerRank Data Structures**
   - https://www.hackerrank.com/domains/data-structures
   - Coding challenges and competitions

*This project demonstrates fundamental computer science concepts through practical implementation and analysis, bridging the gap between theoretical knowledge and real-world application.*