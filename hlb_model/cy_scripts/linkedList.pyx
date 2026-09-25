# linkedList.pyx

import numpy as np
cimport numpy as np
from libc.math cimport INFINITY, NAN

# FIFO: First In - First out. If two events are at the same time, the one that was inserted first takes place first

cdef class EventNode:
    """ Class for each event """
    def __init__(self, np.float64_t time, np.int8_t event_type, np.ndarray cells = np.empty(1, dtype = np.uint8)):
        self.time = time
        self.event_type = event_type
        self.cells = cells
        self.next = None

cdef class LinkedList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.current_size = 0

    cpdef void reset(self):
        """ 
        Clears the linked list and releases memory.
        """
        cdef EventNode current = self.head
        cdef EventNode next_node

        # Traverse and delete all nodes
        while current is not None:
            next_node = current.next
            del current
            current = next_node

        # Reset attributes
        self.head = None
        self.tail = None
        self.current_size = 0

    cpdef void insert(self, np.float64_t time, np.int8_t event_type, np.ndarray cells = np.empty(1, dtype = np.uint8)):
        """ 
        Insertion with tail tracking and last-element fast path
        """
        cdef EventNode new_node = EventNode(time, event_type, cells)
        
        # Empty list case
        if self.head is None:
            self.head = new_node
            self.tail = new_node
            self.current_size += 1
            return

        # Fast path: if new event goes at the end
        if time >= self.tail.time:
            self.tail.next = new_node
            self.tail = new_node
            self.current_size += 1
            return

        # Fast path: if new event goes at the beginning
        if time < self.head.time:
            new_node.next = self.head
            self.head = new_node
            self.current_size += 1
            return

        # General case: find insertion point
        cdef EventNode current = self.head.next
        cdef EventNode prev = self.head
        while current is not None and current.time <= time:
            prev = current
            current = current.next

        # Insert between prev and current
        new_node.next = current
        prev.next = new_node
        self.current_size += 1

    cpdef EventNode pop_top(self):
        """ 
        Remove and return the top (first) element 
        """
        if self.head is None:
            return EventNode(time=-1, event_type=-1, cells=np.empty(1, dtype=np.uint8))

        cdef EventNode top_event = self.head
        self.head = self.head.next
        self.current_size -= 1
        return top_event

    cpdef EventNode find_next_event(self, np.float64_t current_time):
        """ Find the next event that is scheduled before or at current_time, returns it and removes it from the list. """
        cdef:
            cdef EventNode event
            np.float64_t next_t
        
        if self.head is None:
            return EventNode(time=-1, event_type=-1, cells=np.empty(1, dtype=np.uint8))

        next_t = self.peek_time()
        
        if next_t <= current_time:
            return self.pop_top()
        
        return EventNode(time=-1, event_type=-1, cells=np.empty(1, dtype=np.uint8))

    cpdef list pop_top_events(self, set event_types):
        """ 
        Remove and return all top events matching given event types
        """
        cdef list matching_events = []
        
        # If no head or head time is invalid, return empty list
        if self.head is None:
            return matching_events

        # Store the time of top events to compare against
        cdef np.float64_t top_time = self.head.time

        # Temporary head to track list modifications
        cdef EventNode current = self.head
        cdef EventNode prev = None
        
        while current is not None and current.time == top_time:
            # Check if event type matches
            if current.event_type in event_types:
                # Remove this node
                if prev is None:
                    # Removing head
                    self.head = current.next
                    matching_events.append(current)
                    current = self.head
                else:
                    # Removing non-head node
                    prev.next = current.next
                    matching_events.append(current)
                    current = prev.next
                
                self.current_size -= 1
            else:
                # Move to next node
                prev = current
                current = current.next

        return matching_events

    cpdef void pop_next_of_type(self, np.int8_t event_type, np.int16_t n):
        """ 
        Remove next n events of a given type from the list.
        """
        cdef EventNode current = self.head
        cdef EventNode prev = None
        cdef int count = 0
        
        while current is not None and count < n:
            if current.event_type == event_type:
                count += 1
                if prev is None:
                    # Removing head
                    self.head = current.next
                else:
                    # Removing non-head node
                    prev.next = current.next
                
                # If the removed node was the tail, update tail
                if current == self.tail:
                    self.tail = prev
                
                self.current_size -= 1
                current = current.next
            else:
                prev = current
                current = current.next

    cpdef np.float64_t peek_time(self):
        """ Look at the next event time """
        return self.head.time if self.head is not None else INFINITY

    cpdef np.int8_t peek_event_type(self):
        """ Peek at the next event type """
        return self.head.event_type if self.head is not None else -1

    @property
    def size(self):
        """ Return the current size of the list """
        return self.current_size

