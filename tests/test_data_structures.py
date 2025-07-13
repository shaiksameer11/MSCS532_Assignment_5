"""
Unit tests for Elementary Data Structures implementation
Tests correctness, edge cases, and performance characteristics
"""

import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'part2_data_structures'))

from elementary_structures import (
    DynamicArray, Matrix, ArrayStack, ArrayQueue, 
    SinglyLinkedList, RootedTree, TreeNode
)

class TestDynamicArray(unittest.TestCase):
    """Test cases for Dynamic Array implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.arr = DynamicArray()
    
    def test_empty_array(self):
        """Test empty array properties"""
        self.assertEqual(len(self.arr), 0)
        
        with self.assertRaises(IndexError):
            _ = self.arr[0]
    
    def test_append_operation(self):
        """Test append functionality"""
        # Add single element
        self.arr.append(1)
        self.assertEqual(len(self.arr), 1)
        self.assertEqual(self.arr[0], 1)
        
        # Add multiple elements
        for i in range(2, 6):
            self.arr.append(i)
        
        self.assertEqual(len(self.arr), 5)
        for i in range(5):
            self.assertEqual(self.arr[i], i + 1)
    
    def test_insert_operation(self):
        """Test insert functionality"""
        # Insert in empty array
        self.arr.insert(0, 10)
        self.assertEqual(self.arr[0], 10)
        
        # Insert at beginning
        self.arr.insert(0, 5)
        self.assertEqual(self.arr[0], 5)
        self.assertEqual(self.arr[1], 10)
        
        # Insert at end
        self.arr.insert(2, 15)
        self.assertEqual(self.arr[2], 15)
        
        # Insert in middle
        self.arr.insert(1, 7)
        expected = [5, 7, 10, 15]
        for i in range(len(expected)):
            self.assertEqual(self.arr[i], expected[i])
    
    def test_delete_operation(self):
        """Test delete functionality"""
        # Setup array with elements
        for i in range(5):
            self.arr.append(i * 10)
        
        # Delete from middle
        deleted = self.arr.delete(2)
        self.assertEqual(deleted, 20)
        self.assertEqual(len(self.arr), 4)
        
        # Delete from beginning
        deleted = self.arr.delete(0)
        self.assertEqual(deleted, 0)
        
        # Delete from end
        deleted = self.arr.delete(len(self.arr) - 1)
        self.assertEqual(deleted, 40)
    
    def test_index_access(self):
        """Test array indexing"""
        elements = [1, 2, 3, 4, 5]
        for elem in elements:
            self.arr.append(elem)
        
        # Test valid indices
        for i, expected in enumerate(elements):
            self.assertEqual(self.arr[i], expected)
        
        # Test invalid indices
        with self.assertRaises(IndexError):
            _ = self.arr[10]
        
        with self.assertRaises(IndexError):
            _ = self.arr[-1]
    
    def test_automatic_resizing(self):
        """Test automatic array resizing"""
        initial_capacity = self.arr.capacity
        
        # Fill beyond initial capacity
        for i in range(initial_capacity + 5):
            self.arr.append(i)
        
        # Should have resized
        self.assertGreater(self.arr.capacity, initial_capacity)
        self.assertEqual(len(self.arr), initial_capacity + 5)

class TestMatrix(unittest.TestCase):
    """Test cases for Matrix implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.matrix = Matrix(3, 3)
    
    def test_matrix_creation(self):
        """Test matrix initialization"""
        self.assertEqual(self.matrix.rows, 3)
        self.assertEqual(self.matrix.cols, 3)
        
        # Check default values
        for i in range(3):
            for j in range(3):
                self.assertEqual(self.matrix[i, j], 0)
    
    def test_element_access(self):
        """Test matrix element access and modification"""
        # Set elements
        self.matrix[0, 0] = 1
        self.matrix[1, 1] = 2
        self.matrix[2, 2] = 3
        
        # Check elements
        self.assertEqual(self.matrix[0, 0], 1)
        self.assertEqual(self.matrix[1, 1], 2)
        self.assertEqual(self.matrix[2, 2], 3)
        self.assertEqual(self.matrix[0, 1], 0)  # Unchanged element
    
    def test_row_column_access(self):
        """Test row and column access"""
        # Set up test data
        for i in range(3):
            for j in range(3):
                self.matrix[i, j] = i * 3 + j
        
        # Test row access
        row_1 = self.matrix.get_row(1)
        self.assertEqual(row_1, [3, 4, 5])
        
        # Test column access
        col_2 = self.matrix.get_col(2)
        self.assertEqual(col_2, [2, 5, 8])
    
    def test_matrix_addition(self):
        """Test matrix addition"""
        matrix1 = Matrix(2, 2)
        matrix2 = Matrix(2, 2)
        
        # Set up matrices
        matrix1[0, 0] = 1
        matrix1[0, 1] = 2
        matrix1[1, 0] = 3
        matrix1[1, 1] = 4
        
        matrix2[0, 0] = 5
        matrix2[0, 1] = 6
        matrix2[1, 0] = 7
        matrix2[1, 1] = 8
        
        # Add matrices
        result = matrix1.add_matrix(matrix2)
        
        # Check result
        self.assertEqual(result[0, 0], 6)
        self.assertEqual(result[0, 1], 8)
        self.assertEqual(result[1, 0], 10)
        self.assertEqual(result[1, 1], 12)
    
    def test_invalid_operations(self):
        """Test invalid matrix operations"""
        # Invalid index access
        with self.assertRaises(IndexError):
            _ = self.matrix[3, 0]
        
        with self.assertRaises(IndexError):
            self.matrix[0, 3] = 5
        
        # Invalid matrix addition
        other_matrix = Matrix(2, 2)
        with self.assertRaises(ValueError):
            self.matrix.add_matrix(other_matrix)

class TestArrayStack(unittest.TestCase):
    """Test cases for Array-based Stack implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.stack = ArrayStack()
    
    def test_empty_stack(self):
        """Test empty stack properties"""
        self.assertTrue(self.stack.is_empty())
        self.assertEqual(self.stack.size(), 0)
        
        with self.assertRaises(IndexError):
            self.stack.pop()
        
        with self.assertRaises(IndexError):
            self.stack.peek()
    
    def test_push_operation(self):
        """Test push functionality"""
        self.stack.push(10)
        self.assertFalse(self.stack.is_empty())
        self.assertEqual(self.stack.size(), 1)
        self.assertEqual(self.stack.peek(), 10)
        
        self.stack.push(20)
        self.assertEqual(self.stack.size(), 2)
        self.assertEqual(self.stack.peek(), 20)  # Last pushed should be on top
    
    def test_pop_operation(self):
        """Test pop functionality"""
        # Push some elements
        elements = [1, 2, 3, 4, 5]
        for elem in elements:
            self.stack.push(elem)
        
        # Pop elements (should come out in reverse order)
        popped = []
        while not self.stack.is_empty():
            popped.append(self.stack.pop())
        
        self.assertEqual(popped, [5, 4, 3, 2, 1])
    
    def test_peek_operation(self):
        """Test peek functionality"""
        self.stack.push(100)
        self.stack.push(200)
        
        # Peek should not change stack
        top = self.stack.peek()
        self.assertEqual(top, 200)
        self.assertEqual(self.stack.size(), 2)
        
        # Multiple peeks should return same value
        self.assertEqual(self.stack.peek(), 200)
        self.assertEqual(self.stack.peek(), 200)
    
    def test_lifo_behavior(self):
        """Test Last-In-First-Out behavior"""
        items = ['A', 'B', 'C', 'D']
        
        # Push items
        for item in items:
            self.stack.push(item)
        
        # Pop items should be in reverse order
        for expected in reversed(items):
            actual = self.stack.pop()
            self.assertEqual(actual, expected)

class TestArrayQueue(unittest.TestCase):
    """Test cases for Array-based Queue implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.queue = ArrayQueue()
    
    def test_empty_queue(self):
        """Test empty queue properties"""
        self.assertTrue(self.queue.is_empty())
        self.assertEqual(self.queue.get_size(), 0)
        
        with self.assertRaises(IndexError):
            self.queue.dequeue()
        
        with self.assertRaises(IndexError):
            self.queue.peek()
    
    def test_enqueue_operation(self):
        """Test enqueue functionality"""
        self.queue.enqueue(10)
        self.assertFalse(self.queue.is_empty())
        self.assertEqual(self.queue.get_size(), 1)
        self.assertEqual(self.queue.peek(), 10)
        
        self.queue.enqueue(20)
        self.assertEqual(self.queue.get_size(), 2)
        self.assertEqual(self.queue.peek(), 10)  # First enqueued should be at front
    
    def test_dequeue_operation(self):
        """Test dequeue functionality"""
        # Enqueue some elements
        elements = [1, 2, 3, 4, 5]
        for elem in elements:
            self.queue.enqueue(elem)
        
        # Dequeue elements (should come out in same order)
        dequeued = []
        while not self.queue.is_empty():
            dequeued.append(self.queue.dequeue())
        
        self.assertEqual(dequeued, [1, 2, 3, 4, 5])
    
    def test_fifo_behavior(self):
        """Test First-In-First-Out behavior"""
        items = ['X', 'Y', 'Z']
        
        # Enqueue items
        for item in items:
            self.queue.enqueue(item)
        
        # Dequeue items should be in same order
        for expected in items:
            actual = self.queue.dequeue()
            self.assertEqual(actual, expected)
    
    def test_circular_array_behavior(self):
        """Test circular array functionality"""
        # Fill queue to capacity and beyond
        for i in range(10):
            self.queue.enqueue(i)
        
        # Dequeue some elements
        for _ in range(5):
            self.queue.dequeue()
        
        # Enqueue more elements
        for i in range(10, 15):
            self.queue.enqueue(i)
        
        # Check remaining elements are correct
        expected = list(range(5, 15))
        actual = []
        while not self.queue.is_empty():
            actual.append(self.queue.dequeue())
        
        self.assertEqual(actual, expected)

class TestSinglyLinkedList(unittest.TestCase):
    """Test cases for Singly Linked List implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.linked_list = SinglyLinkedList()
    
    def test_empty_list(self):
        """Test empty list properties"""
        self.assertTrue(self.linked_list.is_empty())
        self.assertEqual(self.linked_list.get_size(), 0)
        self.assertEqual(self.linked_list.traverse(), [])
        
        with self.assertRaises(IndexError):
            self.linked_list.delete_at_beginning()
    
    def test_insert_at_beginning(self):
        """Test insertion at beginning"""
        self.linked_list.insert_at_beginning(1)
        self.assertFalse(self.linked_list.is_empty())
        self.assertEqual(self.linked_list.get_size(), 1)
        
        self.linked_list.insert_at_beginning(2)
        self.assertEqual(self.linked_list.traverse(), [2, 1])
    
    def test_insert_at_end(self):
        """Test insertion at end"""
        self.linked_list.insert_at_end(1)
        self.linked_list.insert_at_end(2)
        self.linked_list.insert_at_end(3)
        
        self.assertEqual(self.linked_list.traverse(), [1, 2, 3])
    
    def test_insert_at_position(self):
        """Test insertion at specific position"""
        # Build initial list