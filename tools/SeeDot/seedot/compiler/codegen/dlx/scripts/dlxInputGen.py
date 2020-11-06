#!/usr/bin/env python3

# Program to generate DLX input from memalloc input

import sys
import math

def generateDLXInput(mem_alloc, alignment, max_memory, print_only, dumpFile=None):
    # Read all the memory variables
    mem_vars = []
    num_instructions = 0

    
    for (size, start, end) in mem_alloc:
        size = math.ceil(float(size) / alignment)
        mem_vars.append((size, start, end))
        num_instructions = max(num_instructions, start, end)

    # Identify the minimum amount of memory required
    min_memory_required = 0
    for i in range(1,num_instructions+1):
        inst_size = 0
        for var_size, start, end in mem_vars:
            if start <= i and i <= end:
                inst_size += var_size
        min_memory_required = max(min_memory_required, inst_size)


    min_memory_required = max(min_memory_required, math.floor(float(max_memory) / alignment))

    if print_only:
        return min_memory_required * alignment

    if dumpFile is None:
        assert False, "File for input to DLX solver must be specified"

    # Create the dlx instance
    # Number of columns = number of variables (one for each variable) and
    # number of slots (one for each mem location * instruction)
    num_columns = len(mem_vars) + min_memory_required * num_instructions

    def slot_column_index(i, j):
        return len(mem_vars) + i * num_instructions + j

    tags = []
    subsets = []

    # For each variable add a subset for each possible allocation
    for var_index, (size, start, end) in enumerate(mem_vars):
        mem_start = 0
        mem_end = min_memory_required - size + 1

        # for each placement, add a new subset
        for loc in range(mem_start, mem_end):
            # tag
            tag = "v%d.l%d" % (var_index, loc)
            
            # include the column for this variable
            subset = [var_index]

            # Include all memory locations occupied by the allocation
            for i in range(size):
                for j in range(start-1, end):
                    sci = slot_column_index(loc + i, j)
                    subset.append(sci)

            # add subset
            tags.append(tag)
            subsets.append(subset)
        
        
    outFile = open(dumpFile, "w")

    outFile.write("%d\n"%num_columns)
    outFile.write("%d\n"%len(mem_vars))
    outFile.write("%d\n"%len(subsets))

    for tag, subset in zip(tags, subsets):
        outFile.write("%s %d %s\n" % (tag, len(subset), ' '.join(str(index) for index in subset)))

    outFile.close()
    return None


