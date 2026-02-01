from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

from DAG_kernel_builder import (
    DAGKernelBuilder,
    Instruction
)


class BasicCompiledSolution(DAGKernelBuilder):
    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """        
        # Scratch space addresses
        tmp0 = self.alloc_scratch("tmp0")
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp0, i))
            self.add("load", ("load", self.scratch[v], tmp0))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        
        # vector const registers
        zero_const_v = self.alloc_scratch("zero_const_v", 8)
        one_const_v = self.alloc_scratch("one_const_v", 8)
        two_const_v = self.alloc_scratch("two_const_v", 8)
        self.instrs.append(
            {
                "valu": [
                    ("vbroadcast", zero_const_v, zero_const),
                    ("vbroadcast", one_const_v, one_const), 
                    ("vbroadcast", two_const_v, two_const)
                ]
            }
        )
        
        hash_array1 = self.alloc_scratch("hash_array1", 48)
        hash_array2 = self.alloc_scratch("hash_array2", 48)
        for i in range(len(HASH_STAGES)):
            self.instrs.append(
                {
                    "valu": [
                        ("vbroadcast", hash_array1 + i * 8, self.scratch_const(HASH_STAGES[i][1])),
                        ("vbroadcast", hash_array2 + i * 8, self.scratch_const(HASH_STAGES[i][4]))
                    ]
                }
            )
        
        n_nodes_v = self.alloc_scratch("n_nodes_v", 8)
        self.instrs.append({
            "valu": [("vbroadcast", n_nodes_v, self.scratch["n_nodes"])]
        })
              
        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # TODO: start
        block_size = 16
        vector_size = 8
        
        # vector/scalar scratch registers
        tmp1_v = self.alloc_scratch("tmp1", block_size * vector_size)
        tmp2_v = self.alloc_scratch("tmp2", block_size * vector_size)
        tmp3_v = self.alloc_scratch("tmp3", block_size * vector_size)
        # tmp4_v = self.alloc_scratch("tmp4", block_size * vector_size)
        # tmp5_v = self.alloc_scratch("tmp5", block_size * vector_size)
        # tmp_idx_v = self.alloc_scratch("tmp_idx", block_size * vector_size)
        # tmp_val_v = self.alloc_scratch("tmp_val", block_size * vector_size)
        tmp_node_val_v = self.alloc_scratch("tmp_node_val", block_size * vector_size)
        tmp_addr_v = self.alloc_scratch("tmp_addr", block_size * vector_size)
        
        idx_array = self.alloc_scratch("idx_array", 256)
        val_array = self.alloc_scratch("val_array", 256)
        for i in range(0, 256, 8):
            tmp_addr_1 = tmp_addr_v
            tmp_addr_2 = tmp_addr_v + 1
            self.add_node(Instruction("alu", ("+", tmp_addr_1, self.scratch["inp_indices_p"], self.scratch_const(i)))) 
            self.add_node(Instruction("load", ("vload", idx_array + i, tmp_addr_1)))
            self.add_node(Instruction("alu", ("+", tmp_addr_2, self.scratch["inp_values_p"], self.scratch_const(i))))
            self.add_node(Instruction("load", ("vload", val_array + i, tmp_addr_2)))

        for group_id in range(batch_size // (block_size * vector_size)):
            for block_id in range(block_size):
                self.scratch_const(group_id * block_size * vector_size + block_id * vector_size)
        
        for round in range(rounds):
            for group_id in range(batch_size // (block_size * vector_size)):
                for block_id in range(block_size): # complete block_size * vector_size after loop finishes
                    # each iteration will complete 8 inputs at a time
                    i = group_id * block_size * vector_size + block_id * vector_size
                    
                    # assign block registers
                    tmp1 = tmp1_v + block_id * vector_size
                    tmp2 = tmp2_v + block_id * vector_size
                    tmp3 = tmp3_v + block_id * vector_size
                    # tmp4 = tmp4_v + block_id * vector_size
                    # tmp5 = tmp5_v + block_id * vector_size
                    tmp_node_val = tmp_node_val_v + block_id * vector_size
                    tmp_addr = tmp_addr_v + block_id * vector_size
                    
                    tmp_idx = idx_array + i
                    tmp_val = val_array + i

                    # pull tree nodes
                    for j in range(8):
                        self.add_node(Instruction("alu", ("+", tmp_addr + j, self.scratch["forest_values_p"], tmp_idx + j)))
                        self.add_node(Instruction("load", ("load", tmp_node_val + j, tmp_addr + j)))

                    # val = myhash(val ^ node_val) (vectorized) # TODO: fma the hash?
                    self.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
                    for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                        self.add_node(Instruction("valu", (op1, tmp1, tmp_val, hash_array1 + hi * vector_size)))
                        self.add_node(Instruction("valu", (op3, tmp2, tmp_val, hash_array2 + hi * vector_size)))
                        self.add_node(Instruction("valu", (op2, tmp_val, tmp1, tmp2)))

                    # idx = 2*idx + (1 if val % 2 == 0 else 2) # TODO: not needed on 10th iteration
                    self.add_node(Instruction("valu", ("%", tmp1, tmp_val, two_const_v)))
                    self.add_node(Instruction("valu", ("==", tmp1, tmp1, zero_const_v)))
                    self.add_node(Instruction("flow", ("vselect", tmp3, tmp1, one_const_v, two_const_v)))
                    self.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, two_const_v, tmp3)))

                    # idx = 0 if idx >= n_nodes else idx # TODO: not needed
                    self.add_node(Instruction("valu", ("<", tmp1, tmp_idx, n_nodes_v)))
                    self.add_node(Instruction("flow", ("vselect", tmp_idx, tmp1, tmp_idx, zero_const_v)))
                    
        for i in range(0, 256, 8):
            tmp_addr_1 = tmp_addr_v
            tmp_addr_2 = tmp_addr_v + 1
            self.add_node(Instruction("alu", ("+", tmp_addr_1, self.scratch["inp_indices_p"], self.scratch_const(i)))) 
            self.add_node(Instruction("store", ("vstore", tmp_addr_1, idx_array + i)))
            self.add_node(Instruction("alu", ("+", tmp_addr_2, self.scratch["inp_values_p"], self.scratch_const(i))))
            self.add_node(Instruction("store", ("vstore", tmp_addr_2, val_array + i)))

        self.compile_kernel()
        
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})