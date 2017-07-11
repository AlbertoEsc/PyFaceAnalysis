#! /usr/bin/env python
# benchmarking: This module defines the Benchmark class, which is useful to determine how long
# different sections of the program take to execute. 
#
# By Alberto Escalante. Alberto.Escalante@ini.ruhr-uni-bochum.de
# Ruhr-University-Bochum, Institute of Neural Computation, Group of Prof. Dr. Wiskott

import time
from operator import itemgetter

class Benchmark(object):
    """ A benchmark object allows keeping track of how long different program sections take.
    It is useful to optimize the speed of the code.
    Original code contributed by Alberto N. Escalante-B. (alberto.escalante@ini.rub.de). 
    """    
    def __init__(self, enabled = True):
        """ Initializes the Benchmark object.

        Args:    
            enabled (bool): Determines whether the Benchmark object is enabled. If it is disabled, all member functions
                            do nothing during execution.                                        

        Returns:
            (Benchmark): The newly created Benchmark object.
        """
        self.enabled = enabled
        self.previous_time = {}
        self.previous_time["a"] = time.time()
        self.tasks_dict = {} # keys are text labels, entries are tuples (total_time, num_repetitions, task_number)         
        self.task_number = 0
        self.default_reference = "a"
        # super(Benchmark, self).__init__()
        # def add_reference(self, reference):
        # if reference in self.previous_time.keys():
        #    er = "reference " + str(reference) + " has already been registered, use a different one"
        #    raise Exception (er)
        #    self.previous_time[reference] = time.time()        
    
    def add_task_ellapsed(self, task_label, ellapsed_time, reference=None):
        """ Adds a time measurement to a given task given a time reference.

        Args:    
            task_label (string)): The name of the task. E.g., "Data pre-processing".
            ellapsed_time (float): The time (in seconds) that this task took to execute.
            reference (string): This string identifies the timer used to store the task. If not specified (None), 
                                the default reference is timer "a".
        """
        if not self.enabled:
            return
        if reference is None:
            reference = self.default_reference
        if (reference, task_label) in self.tasks_dict:
            (total_time, num_repetitions, task_number) = self.tasks_dict[(reference, task_label)]
            self.tasks_dict[(reference, task_label)] = (total_time + ellapsed_time, num_repetitions + 1, task_number)
        else:
            self.tasks_dict[(reference, task_label)] = (ellapsed_time, 1, self.task_number)
            self.task_number += 1
    
    def add_task_from_previous_time(self, task_label, reference=None):
        """ Creates a time measurement for a given task with respect to the previous time measurement done 
        using a given time reference.

        Args:    
            task_label (string)): The name of the task. E.g., "Data pre-processing".
            reference (string): This string identifies the timer used to store the task. If not specified (None), 
                                the default reference is timer "a".
        """
        if not self.enabled:
            return
        if reference is None:
            reference = self.default_reference
        now = time.time()
        ellapsed_time = now - self.previous_time[reference]
        self.add_task_ellapsed(task_label, ellapsed_time, reference)
        self.previous_time[reference] = now
    
    def update_start_time(self, reference=None):
        """ The start time of the timer specified by reference is reset to the current time.
        Args:    
            reference (string): This string identifies the timer used to store the task. If not specified (None), 
                                the default reference is timer "a".
        """
        if self.enabled:
            if reference is None:
                reference = self.default_reference
            self.previous_time[reference] = time.time()
    
    def set_default_reference(self, reference):
        """ Changes the reference used by default.
        Args:    
            reference (string): This string identifies the new default timer.
        """
        if self.enabled:
            self.default_reference = reference
    
    def display(self):
        """ Shows (stdout) a summary with all the tasks. 
        
        The information displayed is the reference timers, the task names, the total execution time, the number of executions, and the average execution time. 
        """
        benchmark_array = self.tasks_dict.items() # [(reference, task_label), (total_ellapsed_time, num_repetitions, task_number), ...]
        print "benchmark_array=", benchmark_array
        sorted_benchmark_array = sorted(benchmark_array, key=lambda x: x[1][2])
        
        print "Reference Task" + " "*56 + "  |  avg_time | total_ellapsed_time | num_repetitions"
        for ((reference, task_label), (total_ellapsed_time, num_repetitions, task_number)) in sorted_benchmark_array:
            #(total_ellapsed_time, num_repetitions) = self.tasks_dict[task_label]
            print reference + " "*max(0, 11-len(reference)) + task_label + " "*max(0, 60-len(task_label)) + " |   %07.4f |             %07.4f |      %07d "%(total_ellapsed_time/num_repetitions, total_ellapsed_time, num_repetitions)
        
            
if __name__ == "__main__":
    # This code is useful only to test the module
    import numpy
    benchmark = Benchmark()
    x = numpy.random.normal(size=(1000,1000))
    benchmark.add_task_from_previous_time("generation of x", reference="a")
    benchmark.update_start_time(reference="b")
    benchmark.update_start_time(reference="c")
    benchmark.set_default_reference("c")
    for i in range(10):
        x += numpy.random.uniform(size=(1000,1000))
        benchmark.add_task_from_previous_time("iterative noise addition", reference="b")
    benchmark.add_task_from_previous_time("full noise addition", reference="a")
    benchmark.add_task_from_previous_time("full noise addition")
    x -= x.mean() 
    rep = 20
    benchmark.update_start_time(reference="c")
    for i in range(rep):
        y = numpy.abs(x)**0.8
    benchmark.add_task_from_previous_time("|x|^0.8")
    x32 = x.astype("float32")
    benchmark.update_start_time(reference="c")
    for i in range(rep):
        y = numpy.abs(x32)**0.8
    benchmark.add_task_from_previous_time("|x32|^0.8")
    for i in range(rep):
        y = numpy.power(numpy.abs(x32), 0.8)
    benchmark.add_task_from_previous_time("p|x32|,08")
    print y.dtype
    
    benchmark.display()
    