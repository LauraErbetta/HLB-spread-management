# linkedList.pyx
# language_level=3, boundscheck=True, wraparound=True, initializedcheck=True, cdivision=True

cimport numpy as np

cdef class EventNode:
    cdef:
        public np.float64_t time                  # event time
        public np.int8_t event_type               # event type identifier
        public np.ndarray cells          # cells affected by event, if known
        public EventNode next

cdef class LinkedList:
    cdef:
        EventNode head  # head tracking
        EventNode tail  # tail tracking
        np.int16_t current_size

    cpdef void reset(self)
    cpdef void insert(self, np.float64_t time, np.int8_t event_type, np.ndarray cells =*)
    cpdef EventNode pop_top(self)
    cpdef EventNode find_next_event(self, np.float64_t current_time)
    cpdef list pop_top_events(self, set event_types)
    cpdef void pop_next_of_type(self, np.int8_t event_type, np.int16_t n)
    cpdef np.float64_t peek_time(self)
    cpdef np.int8_t peek_event_type(self)
