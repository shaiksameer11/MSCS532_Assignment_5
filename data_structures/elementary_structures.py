import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Optional, List

class DynamicArray:
    """
    Dynamic Array implementation with automatic resizing
    Supports basic operations like insertion, deletion, and access
    """
    
    def __init__(self, initial_capacity: int = 4):
        """Initialize dynamic array with given capacity"""
        self.capacity = initial_capacity
        self.size = 0
        self.data = [None] * self.capacity
        self.operations_count = 0
    
    def __len__(self) -> int:
        """Return current size of array"""
        return self.size
    
    def __getitem__(self, index: int) -> Any:
        """Access element at given index"""
        self.operations_count += 1
        if not 0 <= index < self.size:
            raise IndexError("Array index out of range")
        return self.data[index]
    
    def __setitem__(self, index: int, value: Any):
        """Set element at given index"""
        self.operations_count += 1
        if not 0 <= index < self.size:
            raise IndexError("Array index out of range")
        self.data[index] = value
    
    def append(self, value: Any):
        """Add element to end of array"""
        self.operations_count += 1
        if self.size == self.capacity:
            self._resize()
        
        self.data[self.size] = value
        self.size += 1
    
    def insert(self, index: int, value: Any):
        """Insert element at given index"""
        self.operations_count += 1
        if not 0 <= index <= self.size:
            raise IndexError("Array index out of range")
        
        if self.size == self.capacity:
            self._resize()
        
        # Shift elements to right
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]
        
        self.data[index] = value
        self.size += 1
    
    def delete(self, index: int) -> Any:
        """Delete and return element at given index"""
        self.operations_count += 1
        if not 0 <= index < self.size:
            raise IndexError("Array index out of range")
        
        deleted_value = self.data[index]
        
        # Shift elements to left
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]
        
        self.size -= 1
        
        # Shrink array if needed
        if self.size <= self.capacity // 4 and self.capacity > 4:
            self._resize(shrink=True)
        
        return deleted_value
    
    def _resize(self, shrink: bool = False):
        """Resize array when needed"""
        if shrink:
            new_capacity = self.capacity // 2
        else:
            new_capacity = self.capacity * 2
        
        new_data = [None] * new_capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        
        self.data = new_data
        self.capacity = new_capacity
    
    def display(self):
        """Display array contents"""
        elements = [self.data[i] for i in range(self.size)]
        print(f"Array: {elements} (size: {self.size}, capacity: {self.capacity})")

class Matrix:
    """
    2D Matrix implementation using nested arrays
    Supports basic matrix operations
    """
    
    def __init__(self, rows: int, cols: int, default_value: Any = 0):
        """Initialize matrix with given dimensions"""
        self.rows = rows
        self.cols = cols
        self.data = [[default_value for _ in range(cols)] for _ in range(rows)]
        self.operations_count = 0
    
    def __getitem__(self, position: tuple) -> Any:
        """Get element at (row, col) position"""
        self.operations_count += 1
        row, col = position
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Matrix index out of range")
        return self.data[row][col]
    
    def __setitem__(self, position: tuple, value: Any):
        """Set element at (row, col) position"""
        self.operations_count += 1
        row, col = position
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError("Matrix index out of range")
        self.data[row][col] = value
    
    def get_row(self, row: int) -> List[Any]:
        """Get entire row"""
        self.operations_count += 1
        if not 0 <= row < self.rows:
            raise IndexError("Row index out of range")
        return self.data[row].copy()
    
    def get_col(self, col: int) -> List[Any]:
        """Get entire column"""
        self.operations_count += 1
        if not 0 <= col < self.cols:
            raise IndexError("Column index out of range")
        return [self.data[row][col] for row in range(self.rows)]
    
    def add_matrix(self, other_matrix: 'Matrix') -> 'Matrix':
        """Add two matrices"""
        if self.rows != other_matrix.rows or self.cols != other_matrix.cols:
            raise ValueError("Matrix dimensions must match for addition")
        
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] + other_matrix[i, j]
        
        return result
    
    def display(self):
        """Display matrix contents"""
        print("Matrix:")
        for row in self.data:
            print("  ", row)

class ArrayStack:
    """
    Stack implementation using dynamic array
    Follows LIFO (Last In, First Out) principle
    """
    
    def __init__(self, initial_capacity: int = 4):
        """Initialize stack with given capacity"""
        self.data = DynamicArray(initial_capacity)
        self.operations_count = 0
    
    def push(self, item: Any):
        """Add item to top of stack"""
        self.operations_count += 1
        self.data.append(item)
    
    def pop(self) -> Any:
        """Remove and return top item from stack"""
        self.operations_count += 1
        if self.is_empty():
            raise IndexError("Cannot pop from empty stack")
        return self.data.delete(len(self.data) - 1)
    
    def peek(self) -> Any:
        """Return top item without removing it"""
        self.operations_count += 1
        if self.is_empty():
            raise IndexError("Cannot peek at empty stack")
        return self.data[len(self.data) - 1]
    
    def is_empty(self) -> bool:
        """Check if stack is empty"""
        return len(self.data) == 0
    
    def size(self) -> int:
        """Get number of items in stack"""
        return len(self.data)
    
    def display(self):
        """Display stack contents"""
        if self.is_empty():
            print("Stack: [] (empty)")
        else:
            elements = [self.data[i] for i in range(len(self.data))]
            print(f"Stack: {elements} (top -> bottom)")

class ArrayQueue:
    """
    Queue implementation using circular array
    Follows FIFO (First In, First Out) principle
    """
    
    def __init__(self, initial_capacity: int = 4):
        """Initialize queue with given capacity"""
        self.capacity = initial_capacity
        self.data = [None] * self.capacity
        self.front = 0
        self.rear = 0
        self.size = 0
        self.operations_count = 0
    
    def enqueue(self, item: Any):
        """Add item to rear of queue"""
        self.operations_count += 1
        if self.size == self.capacity:
            self._resize()
        
        self.data[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
    
    def dequeue(self) -> Any:
        """Remove and return item from front of queue"""
        self.operations_count += 1
        if self.is_empty():
            raise IndexError("Cannot dequeue from empty queue")
        
        item = self.data[self.front]
        self.data[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        
        return item
    
    def peek(self) -> Any:
        """Return front item without removing it"""
        self.operations_count += 1
        if self.is_empty():
            raise IndexError("Cannot peek at empty queue")
        return self.data[self.front]
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self.size == 0
    
    def get_size(self) -> int:
        """Get number of items in queue"""
        return self.size
    
    def _resize(self):
        """Resize queue when capacity is reached"""
        old_data = self.data
        self.capacity *= 2
        self.data = [None] * self.capacity
        
        # Copy elements in correct order
        for i in range(self.size):
            self.data[i] = old_data[(self.front + i) % len(old_data)]
        
        self.front = 0
        self.rear = self.size
    
    def display(self):
        """Display queue contents"""
        if self.is_empty():
            print("Queue: [] (empty)")
        else:
            elements = []
            for i in range(self.size):
                elements.append(self.data[(self.front + i) % self.capacity])
            print(f"Queue: {elements} (front -> rear)")

class ListNode:
    """
    Node class for linked list implementation
    """
    
    def __init__(self, data: Any):
        """Initialize node with data"""
        self.data = data
        self.next: Optional['ListNode'] = None
    
    def __str__(self) -> str:
        return str(self.data)

class SinglyLinkedList:
    """
    Singly Linked List implementation
    Each node points to the next node
    """
    
    def __init__(self):
        """Initialize empty linked list"""
        self.head: Optional[ListNode] = None
        self.size = 0
        self.operations_count = 0
    
    def insert_at_beginning(self, data: Any):
        """Insert new node at beginning of list"""
        self.operations_count += 1
        new_node = ListNode(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1
    
    def insert_at_end(self, data: Any):
        """Insert new node at end of list"""
        self.operations_count += 1
        new_node = ListNode(data)
        
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        
        self.size += 1
    
    def insert_at_position(self, position: int, data: Any):
        """Insert new node at given position"""
        self.operations_count += 1
        if position < 0 or position > self.size:
            raise IndexError("Position out of range")
        
        if position == 0:
            self.insert_at_beginning(data)
            return
        
        new_node = ListNode(data)
        current = self.head
        
        for _ in range(position - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
    
    def delete_at_beginning(self) -> Any:
        """Delete and return first node data"""
        self.operations_count += 1
        if not self.head:
            raise IndexError("Cannot delete from empty list")
        
        data = self.head.data
        self.head = self.head.next
        self.size -= 1
        return data
    
    def delete_at_end(self) -> Any:
        """Delete and return last node data"""
        self.operations_count += 1
        if not self.head:
            raise IndexError("Cannot delete from empty list")
        
        if not self.head.next:
            data = self.head.data
            self.head = None
            self.size -= 1
            return data
        
        current = self.head
        while current.next.next:
            current = current.next
        
        data = current.next.data
        current.next = None
        self.size -= 1
        return data
    
    def delete_at_position(self, position: int) -> Any:
        """Delete and return node data at given position"""
        self.operations_count += 1
        if position < 0 or position >= self.size:
            raise IndexError("Position out of range")
        
        if position == 0:
            return self.delete_at_beginning()
        
        current = self.head
        for _ in range(position - 1):
            current = current.next
        
        data = current.next.data
        current.next = current.next.next
        self.size -= 1
        return data
    
    def search(self, data: Any) -> int:
        """Search for data and return position (-1 if not found)"""
        self.operations_count += 1
        current = self.head
        position = 0
        
        while current:
            if current.data == data:
                return position
            current = current.next
            position += 1
        
        return -1
    
    def get_at_position(self, position: int) -> Any:
        """Get data at given position"""
        self.operations_count += 1
        if position < 0 or position >= self.size:
            raise IndexError("Position out of range")
        
        current = self.head
        for _ in range(position):
            current = current.next
        
        return current.data
    
    def is_empty(self) -> bool:
        """Check if list is empty"""
        return self.head is None
    
    def get_size(self) -> int:
        """Get number of nodes in list"""
        return self.size
    
    def traverse(self) -> List[Any]:
        """Return list of all elements"""
        self.operations_count += 1
        elements = []
        current = self.head
        
        while current:
            elements.append(current.data)
            current = current.next
        
        return elements
    
    def display(self):
        """Display linked list contents"""
        elements = self.traverse()
        if not elements:
            print("Linked List: [] (empty)")
        else:
            arrow_str = " -> ".join(map(str, elements))
            print(f"Linked List: {arrow_str} -> None")

class TreeNode:
    """
    Node class for rooted tree implementation
    """
    
    def __init__(self, data: Any):
        """Initialize tree node with data"""
        self.data = data
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None
    
    def add_child(self, child: 'TreeNode'):
        """Add child node"""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child: 'TreeNode'):
        """Remove child node"""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
    
    def is_leaf(self) -> bool:
        """Check if node is leaf (no children)"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if node is root (no parent)"""
        return self.parent is None

class RootedTree:
    """
    Rooted Tree implementation using linked nodes
    Each node can have multiple children
    """
    
    def __init__(self, root_data: Any = None):
        """Initialize tree with optional root"""
        self.root: Optional[TreeNode] = None
        self.size = 0
        self.operations_count = 0
        
        if root_data is not None:
            self.root = TreeNode(root_data)
            self.size = 1
    
    def insert_child(self, parent_data: Any, child_data: Any) -> bool:
        """Insert child node under parent with given data"""
        self.operations_count += 1
        parent_node = self._find_node(parent_data)
        
        if parent_node:
            child_node = TreeNode(child_data)
            parent_node.add_child(child_node)
            self.size += 1
            return True
        
        return False
    
    def delete_node(self, data: Any) -> bool:
        """Delete node and all its descendants"""
        self.operations_count += 1
        node = self._find_node(data)
        
        if not node:
            return False
        
        if node.is_root():
            self.root = None
            self.size = 0
        else:
            # Count nodes to be deleted
            deleted_count = self._count_subtree_nodes(node)
            node.parent.remove_child(node)
            self.size -= deleted_count
        
        return True
    
    def search(self, data: Any) -> bool:
        """Search for node with given data"""
        self.operations_count += 1
        return self._find_node(data) is not None
    
    def _find_node(self, data: Any) -> Optional[TreeNode]:
        """Find node with given data using DFS"""
        if not self.root:
            return None
        
        return self._dfs_find(self.root, data)
    
    def _dfs_find(self, node: TreeNode, data: Any) -> Optional[TreeNode]:
        """Depth-first search to find node"""
        if node.data == data:
            return node
        
        for child in node.children:
            result = self._dfs_find(child, data)
            if result:
                return result
        
        return None
    
    def _count_subtree_nodes(self, node: TreeNode) -> int:
        """Count nodes in subtree rooted at given node"""
        count = 1  # Count current node
        for child in node.children:
            count += self._count_subtree_nodes(child)
        return count
    
    def preorder_traversal(self) -> List[Any]:
        """Preorder traversal: root -> children"""
        self.operations_count += 1
        result = []
        if self.root:
            self._preorder_helper(self.root, result)
        return result
    
    def _preorder_helper(self, node: TreeNode, result: List[Any]):
        """Helper for preorder traversal"""
        result.append(node.data)
        for child in node.children:
            self._preorder_helper(child, result)
    
    def postorder_traversal(self) -> List[Any]:
        """Postorder traversal: children -> root"""
        self.operations_count += 1
        result = []
        if self.root:
            self._postorder_helper(self.root, result)
        return result
    
    def _postorder_helper(self, node: TreeNode, result: List[Any]):
        """Helper for postorder traversal"""
        for child in node.children:
            self._postorder_helper(child, result)
        result.append(node.data)
    
    def level_order_traversal(self) -> List[Any]:
        """Level order traversal using queue"""
        self.operations_count += 1
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            result.append(node.data)
            
            for child in node.children:
                queue.append(child)
        
        return result
    
    def get_height(self) -> int:
        """Get height of tree"""
        if not self.root:
            return 0
        return self._get_node_height(self.root)
    
    def _get_node_height(self, node: TreeNode) -> int:
        """Get height of subtree rooted at node"""
        if node.is_leaf():
            return 1
        
        max_child_height = 0
        for child in node.children:
            child_height = self._get_node_height(child)
            max_child_height = max(max_child_height, child_height)
        
        return max_child_height + 1
    
    def get_size(self) -> int:
        """Get number of nodes in tree"""
        return self.size
    
    def is_empty(self) -> bool:
        """Check if tree is empty"""
        return self.root is None
    
    def display(self):
        """Display tree structure"""
        if self.is_empty():
            print("Tree: (empty)")
        else:
            print("Tree structure:")
            self._display_helper(self.root, "", True)
    
    def _display_helper(self, node: TreeNode, prefix: str, is_last: bool):
        """Helper for displaying tree structure"""
        if node:
            print(prefix + ("|-- " if is_last else "+-- ") + str(node.data))
            
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                extension = "    " if is_last else "|   "
                self._display_helper(child, prefix + extension, is_last_child)

class DataStructureAnalyzer:
    """
    Analyzer for comparing performance of different data structures
    """
    
    def benchmark_operations(self, sizes: List[int]) -> dict:
        """
        Benchmark basic operations on different data structures
        
        Parameters:
            sizes: List of sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'sizes': sizes,
            'array_insert': [],
            'array_delete': [],
            'array_access': [],
            'list_insert': [],
            'list_delete': [],
            'list_search': [],
            'stack_push': [],
            'stack_pop': [],
            'queue_enqueue': [],
            'queue_dequeue': []
        }
        
        for size in sizes:
            print(f"Benchmarking with size {size}...")
            
            # Benchmark Dynamic Array
            arr = DynamicArray()
            
            # Array insertions
            start_time = time.perf_counter()
            for i in range(size):
                arr.append(i)
            results['array_insert'].append(time.perf_counter() - start_time)
            
            # Array access
            start_time = time.perf_counter()
            for i in range(min(1000, size)):
                _ = arr[i % size]
            results['array_access'].append(time.perf_counter() - start_time)
            
            # Array deletions
            start_time = time.perf_counter()
            for _ in range(min(100, size // 2)):
                if len(arr) > 0:
                    arr.delete(0)
            results['array_delete'].append(time.perf_counter() - start_time)
            
            # Benchmark Linked List
            linked_list = SinglyLinkedList()
            
            # List insertions
            start_time = time.perf_counter()
            for i in range(size):
                linked_list.insert_at_end(i)
            results['list_insert'].append(time.perf_counter() - start_time)
            
            # List search
            start_time = time.perf_counter()
            for i in range(min(100, size)):
                linked_list.search(i)
            results['list_search'].append(time.perf_counter() - start_time)
            
            # List deletions
            start_time = time.perf_counter()
            for _ in range(min(100, size // 2)):
                if not linked_list.is_empty():
                    linked_list.delete_at_beginning()
            results['list_delete'].append(time.perf_counter() - start_time)
            
            # Benchmark Stack
            stack = ArrayStack()
            
            # Stack push
            start_time = time.perf_counter()
            for i in range(size):
                stack.push(i)
            results['stack_push'].append(time.perf_counter() - start_time)
            
            # Stack pop
            start_time = time.perf_counter()
            for _ in range(min(100, size)):
                if not stack.is_empty():
                    stack.pop()
            results['stack_pop'].append(time.perf_counter() - start_time)
            
            # Benchmark Queue
            queue = ArrayQueue()
            
            # Queue enqueue
            start_time = time.perf_counter()
            for i in range(size):
                queue.enqueue(i)
            results['queue_enqueue'].append(time.perf_counter() - start_time)
            
            # Queue dequeue
            start_time = time.perf_counter()
            for _ in range(min(100, size)):
                if not queue.is_empty():
                    queue.dequeue()
            results['queue_dequeue'].append(time.perf_counter() - start_time)
        
        return results
    
    def plot_performance_results(self, results: dict):
        """Create graphs showing performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        sizes = results['sizes']
        
        # Graph 1: Insert Operations
        ax1.plot(sizes, results['array_insert'], 'o-', label='Array Insert', linewidth=2)
        ax1.plot(sizes, results['list_insert'], 's-', label='Linked List Insert', linewidth=2)
        ax1.plot(sizes, results['stack_push'], '^-', label='Stack Push', linewidth=2)
        ax1.plot(sizes, results['queue_enqueue'], 'd-', label='Queue Enqueue', linewidth=2)
        ax1.set_xlabel('Data Structure Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Insert Operations Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # Graph 2: Delete Operations
        ax2.plot(sizes, results['array_delete'], 'o-', label='Array Delete', linewidth=2)
        ax2.plot(sizes, results['list_delete'], 's-', label='Linked List Delete', linewidth=2)
        ax2.plot(sizes, results['stack_pop'], '^-', label='Stack Pop', linewidth=2)
        ax2.plot(sizes, results['queue_dequeue'], 'd-', label='Queue Dequeue', linewidth=2)
        ax2.set_xlabel('Data Structure Size')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Delete Operations Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Graph 3: Access Operations
        ax3.plot(sizes, results['array_access'], 'o-', label='Array Access', linewidth=2)
        ax3.plot(sizes, results['list_search'], 's-', label='Linked List Search', linewidth=2)
        ax3.set_xlabel('Data Structure Size')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Access/Search Operations Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        # Graph 4: Combined Operations Comparison
        ax4.plot(sizes, results['array_insert'], 'o-', label='Array Operations', linewidth=2)
        ax4.plot(sizes, results['list_insert'], 's-', label='Linked List Operations', linewidth=2)
        ax4.set_xlabel('Data Structure Size')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Overall Performance Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('data_structures_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_complexity(self):
        """
        Analyze time complexity of different data structure operations
        """
        print("=== DATA STRUCTURES TIME COMPLEXITY ANALYSIS ===\n")
        
        print("1. Dynamic Array:")
        print("=================")
        print("- Access by index: O(1) - direct array indexing")
        print("- Insert at end: O(1) amortized - may need to resize occasionally")
        print("- Insert at middle: O(n) - need to shift elements")
        print("- Delete at end: O(1) - just reduce size")
        print("- Delete at middle: O(n) - need to shift elements")
        print("- Search: O(n) - need to check each element")
        print()
        
        print("2. Singly Linked List:")
        print("======================")
        print("- Access by position: O(n) - need to traverse from head")
        print("- Insert at beginning: O(1) - just update head pointer")
        print("- Insert at end: O(n) - need to traverse to end")
        print("- Insert at middle: O(n) - need to traverse to position")
        print("- Delete at beginning: O(1) - just update head pointer")
        print("- Delete at end: O(n) - need to traverse to second last")
        print("- Delete at middle: O(n) - need to traverse to position")
        print("- Search: O(n) - need to traverse through nodes")
        print()
        
        print("3. Stack (Array-based):")
        print("========================")
        print("- Push: O(1) amortized - add to end of array")
        print("- Pop: O(1) - remove from end of array")
        print("- Peek: O(1) - look at last element")
        print("- Size check: O(1) - maintain size counter")
        print()
        
        print("4. Queue (Circular Array):")
        print("===========================")
        print("- Enqueue: O(1) amortized - add at rear")
        print("- Dequeue: O(1) - remove from front")
        print("- Peek: O(1) - look at front element")
        print("- Size check: O(1) - maintain size counter")
        print()
        
        print("5. Rooted Tree:")
        print("===============")
        print("- Insert: O(h) where h is height - need to find parent")
        print("- Delete: O(h) - need to find node first")
        print("- Search: O(n) worst case - might need to check all nodes")
        print("- Traversal: O(n) - visit each node once")
        print()
        
        print("Space Complexity:")
        print("=================")
        print("- Dynamic Array: O(n) - store n elements")
        print("- Linked List: O(n) - store n nodes with pointers")
        print("- Stack: O(n) - same as underlying array")
        print("- Queue: O(n) - same as underlying array")
        print("- Tree: O(n) - store n nodes with pointers")

def demonstrate_data_structures():
    """
    Demonstrate basic usage of all implemented data structures
    """
    print("=== DATA STRUCTURES DEMONSTRATION ===\n")
    
    # Dynamic Array Demo
    print("1. Dynamic Array Demo:")
    print("======================")
    arr = DynamicArray()
    print("Creating empty array...")
    arr.display()
    
    print("Adding elements 1, 2, 3...")
    for i in [1, 2, 3]:
        arr.append(i)
    arr.display()
    
    print("Inserting 0 at beginning...")
    arr.insert(0, 0)
    arr.display()
    
    print("Deleting element at index 2...")
    deleted = arr.delete(2)
    print(f"Deleted: {deleted}")
    arr.display()
    print()
    
    # Matrix Demo
    print("2. Matrix Demo:")
    print("===============")
    matrix = Matrix(3, 3)
    print("Creating 3x3 matrix filled with zeros...")
    matrix.display()
    
    print("Setting some values...")
    matrix[0, 0] = 1
    matrix[1, 1] = 2
    matrix[2, 2] = 3
    matrix.display()
    print()
    
    # Stack Demo
    print("3. Stack Demo:")
    print("==============")
    stack = ArrayStack()
    print("Creating empty stack...")
    stack.display()
    
    print("Pushing elements 10, 20, 30...")
    for item in [10, 20, 30]:
        stack.push(item)
        print(f"Pushed {item}")
    stack.display()
    
    print("Popping elements...")
    while not stack.is_empty():
        popped = stack.pop()
        print(f"Popped: {popped}")
    stack.display()
    print()
    
    # Queue Demo
    print("4. Queue Demo:")
    print("==============")
    queue = ArrayQueue()
    print("Creating empty queue...")
    queue.display()
    
    print("Enqueuing elements A, B, C...")
    for item in ['A', 'B', 'C']:
        queue.enqueue(item)
        print(f"Enqueued {item}")
    queue.display()
    
    print("Dequeuing elements...")
    while not queue.is_empty():
        dequeued = queue.dequeue()
        print(f"Dequeued: {dequeued}")
    queue.display()
    print()
    
    # Linked List Demo
    print("5. Linked List Demo:")
    print("====================")
    linked_list = SinglyLinkedList()
    print("Creating empty linked list...")
    linked_list.display()
    
    print("Inserting elements at end: 5, 10, 15...")
    for item in [5, 10, 15]:
        linked_list.insert_at_end(item)
    linked_list.display()
    
    print("Inserting 0 at beginning...")
    linked_list.insert_at_beginning(0)
    linked_list.display()
    
    print("Searching for element 10...")
    position = linked_list.search(10)
    print(f"Element 10 found at position: {position}")
    
    print("Deleting element at position 2...")
    deleted = linked_list.delete_at_position(2)
    print(f"Deleted: {deleted}")
    linked_list.display()
    print()
    
    # Tree Demo
    print("6. Rooted Tree Demo:")
    print("====================")
    tree = RootedTree("Root")
    print("Creating tree with root 'Root'...")
    tree.display()
    
    print("Adding children A and B to Root...")
    tree.insert_child("Root", "A")
    tree.insert_child("Root", "B")
    tree.display()
    
    print("Adding children A1, A2 to A...")
    tree.insert_child("A", "A1")
    tree.insert_child("A", "A2")
    tree.display()
    
    print("Adding child B1 to B...")
    tree.insert_child("B", "B1")
    tree.display()
    
    print("Tree traversals:")
    print(f"Preorder: {tree.preorder_traversal()}")
    print(f"Postorder: {tree.postorder_traversal()}")
    print(f"Level order: {tree.level_order_traversal()}")
    print(f"Tree height: {tree.get_height()}")

# Example usage and testing
if __name__ == "__main__":
    # Demonstrate all data structures
    demonstrate_data_structures()
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Analyze complexity
    analyzer = DataStructureAnalyzer()
    analyzer.analyze_complexity()
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    # Benchmark performance
    sizes = [100, 500, 1000, 2000, 5000]
    results = analyzer.benchmark_operations(sizes)
    
    # Create performance graphs
    analyzer.plot_performance_results(results)
    
    # Print summary
    print("\nPerformance Summary:")
    print("===================")
    print("Based on the benchmarks:")
    print("- Array access is fastest (O(1))")
    print("- Linked list search is slower (O(n))")
    print("- Stack and queue operations are very fast (O(1))")
    print("- Array insertion at end is fast due to amortization")
    print("- Linked list insertion at beginning is very fast")