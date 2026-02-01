from problem import HASH_STAGES

from DAG_kernel_builder import DAGKernelBuilder, Instruction

CHARACTER_COUNT = 256
ROUNDS = 16
BLOCK_SIZE = 8
VECTOR_SIZE = 8


def kernel(kb: DAGKernelBuilder):
    # vector/scalar scratch registers
    tmp1_v = kb.alloc_scratch("tmp1_v", BLOCK_SIZE * VECTOR_SIZE)
    tmp2_v = kb.alloc_scratch("tmp2_v", BLOCK_SIZE * VECTOR_SIZE)
    tmp3_v = kb.alloc_scratch("tmp3_v", BLOCK_SIZE * VECTOR_SIZE)
    tmp4_v = kb.alloc_scratch("tmp4_v", BLOCK_SIZE * VECTOR_SIZE)
    # tmp5_v = kb.alloc_scratch("tmp5_v", BLOCK_SIZE * VECTOR_SIZE)
    # tmp_idx_v = kb.alloc_scratch("tmp_idx", BLOCK_SIZE * VECTOR_SIZE)
    # tmp_val_v = kb.alloc_scratch("tmp_val", BLOCK_SIZE * VECTOR_SIZE)
    tmp_node_val_v = kb.alloc_scratch("tmp_node_val_v", BLOCK_SIZE * VECTOR_SIZE)
    # tmp_addr_v = kb.alloc_scratch("tmp_addr_v", BLOCK_SIZE * VECTOR_SIZE)
    tmp_addr_v = kb.alloc_scratch("tmp_addr_v", 256)

    idx_array = kb.alloc_scratch("idx_array", 256)
    val_array = kb.alloc_scratch("val_array", 256)
    for i in range(0, 256, 8):
        tmp_addr_1 = tmp_addr_v
        tmp_addr_2 = tmp_addr_v + 1
        kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("load", ("vload", idx_array + i, tmp_addr_1)))
        kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("load", ("vload", val_array + i, tmp_addr_2)))

    for round_num in range(ROUNDS):
        round_kernel(kb, round_num)
        # kb.compile_kernel()
        # for i in range(0, 256, 8):
        #     kb.instrs.append({"debug": [("vcompare", idx_array + i, [(round_num, i + j, "wrapped_idx") for j in range(8)])]})

    for i in range(0, 256, 8):
        tmp_addr_1 = tmp_addr_v
        tmp_addr_2 = tmp_addr_v + 1
        kb.add_node(Instruction("alu", ("+", tmp_addr_1, kb.scratch["inp_indices_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_1, idx_array + i)))
        kb.add_node(Instruction("alu", ("+", tmp_addr_2, kb.scratch["inp_values_p"], kb.scratch_const(i))))
        kb.add_node(Instruction("store", ("vstore", tmp_addr_2, val_array + i)))

    kb.compile_kernel()

    # Required to match with the yield in reference_kernel2
    kb.instrs.append({"flow": [("pause",)]})


def round_kernel(kb: DAGKernelBuilder, round_num: int):
    if round_num == 0 or round_num == 11:
        round_0_kernel(kb)
    else:
        generic_round_kernel(kb, round_num)
    
    if round_num == 2:
        kb.compile_kernel()


def generic_round_kernel(kb: DAGKernelBuilder, round_num: int):
    for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * VECTOR_SIZE)):
        for block_id in range(BLOCK_SIZE):  # complete BLOCK_SIZE * VECTOR_SIZE after loop finishes
            # each iteration will complete 8 inputs at a time
            i = group_id * BLOCK_SIZE * VECTOR_SIZE + block_id * VECTOR_SIZE

            # assign block registers
            tmp1 = kb.scratch["tmp1_v"] + block_id * VECTOR_SIZE
            tmp2 = kb.scratch["tmp2_v"] + block_id * VECTOR_SIZE
            tmp3 = kb.scratch["tmp3_v"] + block_id * VECTOR_SIZE
            # tmp4 = tmp4_v + block_id * VECTOR_SIZE
            # tmp5 = tmp5_v + block_id * VECTOR_SIZE
            tmp_node_val = kb.scratch["tmp_node_val_v"] + block_id * VECTOR_SIZE
            tmp_addr = kb.scratch["tmp_addr_v"] + block_id * VECTOR_SIZE

            tmp_idx = kb.scratch["idx_array"] + i
            tmp_val = kb.scratch["val_array"] + i

            # pull tree nodes
            for j in range(8):
                kb.add_node(Instruction("alu", ("+", tmp_addr + j, kb.scratch["forest_values_p"], tmp_idx + j)))
                kb.add_node(Instruction("load", ("load", tmp_node_val + j, tmp_addr + j)))

            # val = myhash(val ^ node_val) (vectorized) # TODO: fma the hash?
            kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, tmp_node_val)))
            for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                kb.add_node(Instruction("valu", (op1, tmp1, tmp_val, kb.scratch["hash_array1"] + hi * VECTOR_SIZE)))
                kb.add_node(Instruction("valu", (op3, tmp2, tmp_val, kb.scratch["hash_array2"] + hi * VECTOR_SIZE)))
                kb.add_node(Instruction("valu", (op2, tmp_val, tmp1, tmp2)))

            if round_num != 10:
                # idx = 2*idx + (1 if val % 2 == 0 else 2) # TODO: not needed on 10th iteration
                kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["two_const_v"])))
                kb.add_node(Instruction("valu", ("==", tmp1, tmp1, kb.scratch["zero_const_v"])))
                kb.add_node(Instruction("flow", ("vselect", tmp3, tmp1, kb.scratch["one_const_v"], kb.scratch["two_const_v"])))
                kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["two_const_v"], tmp3)))
            else:
                # idx = root (0) if on level 10
                kb.add_node(Instruction("valu", ("&", tmp_idx, kb.scratch["zero_const_v"], kb.scratch["zero_const_v"])))


def round_0_kernel(kb: DAGKernelBuilder):
    # pull tree nodes (all nodes are the same root node)
    root_val = kb.scratch["tmp4_v"]
    kb.add_node(Instruction("load", ("load", root_val, kb.scratch["forest_values_p"])))
    kb.add_node(Instruction("valu", ("vbroadcast", root_val, root_val)))

    for group_id in range(CHARACTER_COUNT // (BLOCK_SIZE * VECTOR_SIZE)):
        for block_id in range(BLOCK_SIZE):
            i = group_id * BLOCK_SIZE * VECTOR_SIZE + block_id * VECTOR_SIZE

            # assign block registers
            tmp1 = kb.scratch["tmp1_v"] + block_id * VECTOR_SIZE
            tmp2 = kb.scratch["tmp2_v"] + block_id * VECTOR_SIZE
            tmp3 = kb.scratch["tmp3_v"] + block_id * VECTOR_SIZE

            tmp_idx = kb.scratch["idx_array"] + i
            tmp_val = kb.scratch["val_array"] + i

            # val = myhash(val ^ node_val) (vectorized) # TODO: fma the hash?
            kb.add_node(Instruction("valu", ("^", tmp_val, tmp_val, root_val)))
            for hi, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                kb.add_node(Instruction("valu", (op1, tmp1, tmp_val, kb.scratch["hash_array1"] + hi * VECTOR_SIZE)))
                kb.add_node(Instruction("valu", (op3, tmp2, tmp_val, kb.scratch["hash_array2"] + hi * VECTOR_SIZE)))
                kb.add_node(Instruction("valu", (op2, tmp_val, tmp1, tmp2)))

            # idx = 2*idx + (1 if val % 2 == 0 else 2)
            kb.add_node(Instruction("valu", ("%", tmp1, tmp_val, kb.scratch["two_const_v"])))
            kb.add_node(Instruction("valu", ("==", tmp1, tmp1, kb.scratch["zero_const_v"])))
            kb.add_node(Instruction("flow", ("vselect", tmp3, tmp1, kb.scratch["one_const_v"], kb.scratch["two_const_v"])))
            kb.add_node(Instruction("valu", ("multiply_add", tmp_idx, tmp_idx, kb.scratch["two_const_v"], tmp3)))
