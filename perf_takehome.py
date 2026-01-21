"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

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


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            # Simple slot packing that just uses one slot per instruction bundle
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        def vec_range(base):
            return set(range(base, base + VLEN))

        def slot_reads_writes(engine, slot):
            reads = set()
            writes = set()
            if engine == "alu":
                _, dest, a1, a2 = slot
                reads.update([a1, a2])
                writes.add(dest)
            elif engine == "valu":
                match slot:
                    case ("vbroadcast", dest, src):
                        reads.add(src)
                        writes.update(vec_range(dest))
                    case ("multiply_add", dest, a, b, c):
                        reads.update(vec_range(a))
                        reads.update(vec_range(b))
                        reads.update(vec_range(c))
                        writes.update(vec_range(dest))
                    case (op, dest, a1, a2):
                        reads.update(vec_range(a1))
                        reads.update(vec_range(a2))
                        writes.update(vec_range(dest))
            elif engine == "load":
                match slot:
                    case ("load", dest, addr):
                        reads.add(addr)
                        writes.add(dest)
                    case ("load_offset", dest, addr, offset):
                        reads.add(addr + offset)
                        writes.add(dest + offset)
                    case ("vload", dest, addr):
                        reads.add(addr)
                        writes.update(vec_range(dest))
                    case ("const", dest, _val):
                        writes.add(dest)
            elif engine == "store":
                match slot:
                    case ("store", addr, src):
                        reads.add(addr)
                        reads.add(src)
                    case ("vstore", addr, src):
                        reads.add(addr)
                        reads.update(vec_range(src))
            elif engine == "flow":
                match slot:
                    case ("select", dest, cond, a, b):
                        reads.update([cond, a, b])
                        writes.add(dest)
                    case ("add_imm", dest, a, _imm):
                        reads.add(a)
                        writes.add(dest)
                    case ("vselect", dest, cond, a, b):
                        reads.update(vec_range(cond))
                        reads.update(vec_range(a))
                        reads.update(vec_range(b))
                        writes.update(vec_range(dest))
                    case ("cond_jump", cond, _addr):
                        reads.add(cond)
                    case ("cond_jump_rel", cond, _offset):
                        reads.add(cond)
                    case ("jump", addr):
                        reads.add(addr)
                    case ("jump_indirect", addr):
                        reads.add(addr)
                    case ("trace_write", val):
                        reads.add(val)
                    case ("coreid", dest):
                        writes.add(dest)
                    case ("halt",) | ("pause",):
                        pass
            elif engine == "debug":
                match slot:
                    case ("compare", loc, _key):
                        reads.add(loc)
                    case ("vcompare", loc, _keys):
                        reads.update(vec_range(loc))
                    case ("comment", _msg):
                        pass
            return reads, writes

        const_addr_to_val = {addr: val for val, addr in self.const_map.items()}
        base_addrs = {
            self.scratch.get("inp_indices_p"): "indices",
            self.scratch.get("inp_values_p"): "values",
        }
        expr_map = {}
        nodes = []

        for engine, slot in slots:
            reads, writes = slot_reads_writes(engine, slot)
            mem_reads = set()
            mem_writes = set()

            if engine == "load":
                match slot:
                    case ("load", _dest, addr):
                        token = expr_map.get(addr)
                        if token:
                            mem_reads.add((token[0], token[1], "s"))
                    case ("vload", _dest, addr):
                        token = expr_map.get(addr)
                        if token:
                            mem_reads.add((token[0], token[1], "v"))
            elif engine == "store":
                match slot:
                    case ("store", addr, _src):
                        token = expr_map.get(addr)
                        if token:
                            mem_writes.add((token[0], token[1], "s"))
                    case ("vstore", addr, _src):
                        token = expr_map.get(addr)
                        if token:
                            mem_writes.add((token[0], token[1], "v"))

            nodes.append(
                {
                    "engine": engine,
                    "slot": slot,
                    "reads": reads,
                    "writes": writes,
                    "mem_reads": mem_reads,
                    "mem_writes": mem_writes,
                }
            )

            if engine == "alu":
                op, dest, a1, a2 = slot
                base = None
                offset = None
                if op == "+":
                    if a1 in base_addrs and a2 in const_addr_to_val:
                        base = base_addrs[a1]
                        offset = const_addr_to_val[a2]
                    elif a2 in base_addrs and a1 in const_addr_to_val:
                        base = base_addrs[a2]
                        offset = const_addr_to_val[a1]
                if base is not None:
                    expr_map[dest] = (base, offset)
                else:
                    expr_map.pop(dest, None)
            else:
                for addr in writes:
                    expr_map.pop(addr, None)

        last_write = {}
        last_read = {}
        mem_last_write = {}
        mem_last_read = {}
        deps = [set() for _ in nodes]

        for idx, node in enumerate(nodes):
            for addr in node["reads"]:
                if addr in last_write:
                    if last_write[addr] != idx:
                        deps[idx].add(last_write[addr])
                last_read[addr] = idx
            for addr in node["writes"]:
                if addr in last_write:
                    if last_write[addr] != idx:
                        deps[idx].add(last_write[addr])
                if addr in last_read:
                    if last_read[addr] != idx:
                        deps[idx].add(last_read[addr])
                last_write[addr] = idx
                last_read.pop(addr, None)

            for token in node["mem_reads"]:
                if token in mem_last_write:
                    if mem_last_write[token] != idx:
                        deps[idx].add(mem_last_write[token])
                mem_last_read[token] = idx
            for token in node["mem_writes"]:
                if token in mem_last_write:
                    if mem_last_write[token] != idx:
                        deps[idx].add(mem_last_write[token])
                if token in mem_last_read:
                    if mem_last_read[token] != idx:
                        deps[idx].add(mem_last_read[token])
                mem_last_write[token] = idx
                mem_last_read.pop(token, None)

        succs = [set() for _ in nodes]
        indegree = [0] * len(nodes)
        for idx, dep_set in enumerate(deps):
            for dep in dep_set:
                succs[dep].add(idx)
                indegree[idx] += 1

        ready = [i for i, deg in enumerate(indegree) if deg == 0]
        instrs = []
        scheduled = 0
        while scheduled < len(nodes):
            counts = defaultdict(int)
            bundle = defaultdict(list)
            next_ready = []
            scheduled_this = []
            for idx in ready:
                eng = nodes[idx]["engine"]
                if counts[eng] < SLOT_LIMITS[eng]:
                    bundle[eng].append(nodes[idx]["slot"])
                    counts[eng] += 1
                    scheduled_this.append(idx)
                else:
                    next_ready.append(idx)
            if not scheduled_this:
                raise RuntimeError("Scheduling stalled with no ready slots.")
            instrs.append(dict(bundle))
            scheduled += len(scheduled_this)
            for idx in scheduled_this:
                for succ in succs[idx]:
                    indegree[succ] -= 1
                    if indegree[succ] == 0:
                        next_ready.append(succ)
            ready = next_ready

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, include_debug=False):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            if include_debug:
                slots.append(
                    ("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi)))
                )

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
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
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")
        n_bufs = 4
        tmp_addr_vec = []
        tmp_addr2_vec = []
        for bi in range(n_bufs):
            tmp_addr_vec.append(self.alloc_scratch(f"tmp_addr_vec_{bi}"))
            tmp_addr2_vec.append(self.alloc_scratch(f"tmp_addr2_vec_{bi}"))

        # Vector scratch registers (multi-buffered to increase ILP)
        v_idx = []
        v_val = []
        v_node_val = []
        v_tmp1 = []
        v_tmp2 = []
        v_tmp3 = []
        v_addr = []
        v_cond = []
        for bi in range(n_bufs):
            v_idx.append(self.alloc_scratch(f"v_idx_{bi}", VLEN))
            v_val.append(self.alloc_scratch(f"v_val_{bi}", VLEN))
            v_node_val.append(self.alloc_scratch(f"v_node_val_{bi}", VLEN))
            v_tmp1.append(self.alloc_scratch(f"v_tmp1_{bi}", VLEN))
            v_tmp2.append(self.alloc_scratch(f"v_tmp2_{bi}", VLEN))
            v_tmp3.append(self.alloc_scratch(f"v_tmp3_{bi}", VLEN))
            v_addr.append(self.alloc_scratch(f"v_addr_{bi}", VLEN))
            v_cond.append(self.alloc_scratch(f"v_cond_{bi}", VLEN))

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        body.append(
            (
                "valu",
                ("vbroadcast", v_zero, zero_const),
            )
        )
        body.append(
            (
                "valu",
                ("vbroadcast", v_one, one_const),
            )
        )
        body.append(
            (
                "valu",
                ("vbroadcast", v_two, two_const),
            )
        )
        body.append(
            (
                "valu",
                ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            )
        )
        body.append(
            (
                "valu",
                ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
            )
        )

        # Pre-broadcast hash constants to vectors to avoid per-iteration setup.
        vec_const = {}
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            for val in (val1, val3):
                if val in vec_const:
                    continue
                v_const = self.alloc_scratch(f"v_const_{val}", VLEN)
                vec_const[val] = v_const
                body.append(("valu", ("vbroadcast", v_const, self.scratch_const(val))))
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul = 1 + (1 << val3)
                if mul not in vec_const:
                    v_const = self.alloc_scratch(f"v_const_{mul}", VLEN)
                    vec_const[mul] = v_const
                    body.append(
                        ("valu", ("vbroadcast", v_const, self.scratch_const(mul)))
                    )

        vec_batch = batch_size - (batch_size % VLEN)
        for round in range(rounds):
            for i in range(0, vec_batch, VLEN):
                bi = (i // VLEN) % n_bufs
                v_idx_b = v_idx[bi]
                v_val_b = v_val[bi]
                v_node_val_b = v_node_val[bi]
                v_tmp1_b = v_tmp1[bi]
                v_tmp2_b = v_tmp2[bi]
                v_tmp3_b = v_tmp3[bi]
                v_addr_b = v_addr[bi]
                v_cond_b = v_cond[bi]
                addr_idx_b = tmp_addr_vec[bi]
                addr_val_b = tmp_addr2_vec[bi]
                i_const = self.scratch_const(i)
                # Compute base addresses for vector loads/stores.
                body.append(
                    (
                        "alu",
                        ("+", addr_idx_b, self.scratch["inp_indices_p"], i_const),
                    )
                )
                body.append(
                    (
                        "alu",
                        ("+", addr_val_b, self.scratch["inp_values_p"], i_const),
                    )
                )
                body.append(
                    (
                        "load",
                        ("vload", v_idx_b, addr_idx_b),
                    )
                )
                body.append(
                    (
                        "load",
                        ("vload", v_val_b, addr_val_b),
                    )
                )
                # v_addr = forest_values_p + v_idx
                body.append(("valu", ("+", v_addr_b, v_idx_b, v_forest_p)))
                # Gather node values using per-lane loads.
                for lane in range(VLEN):
                    body.append(("load", ("load_offset", v_node_val_b, v_addr_b, lane)))
                # v_val ^= v_node_val
                body.append(("valu", ("^", v_val_b, v_val_b, v_node_val_b)))

                # Hash stages (vectorized)
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    if op1 == "+" and op2 == "+" and op3 == "<<":
                        mul = 1 + (1 << val3)
                        body.append(
                            (
                                "valu",
                                (
                                    "multiply_add",
                                    v_val_b,
                                    v_val_b,
                                    vec_const[mul],
                                    vec_const[val1],
                                ),
                            )
                        )
                        continue
                    body.append(
                        (
                            "valu",
                            (op1, v_tmp1_b, v_val_b, vec_const[val1]),
                        )
                    )
                    body.append(
                        (
                            "valu",
                            (op3, v_tmp2_b, v_val_b, vec_const[val3]),
                        )
                    )
                    body.append(("valu", (op2, v_val_b, v_tmp1_b, v_tmp2_b)))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("&", v_tmp1_b, v_val_b, v_one)))
                body.append(("flow", ("vselect", v_tmp3_b, v_tmp1_b, v_two, v_one)))
                body.append(("valu", ("*", v_tmp2_b, v_idx_b, v_two)))
                body.append(("valu", ("+", v_idx_b, v_tmp2_b, v_tmp3_b)))

                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", v_cond_b, v_idx_b, v_n_nodes)))
                body.append(("flow", ("vselect", v_idx_b, v_cond_b, v_idx_b, v_zero)))

                body.append(("store", ("vstore", addr_idx_b, v_idx_b)))
                body.append(("store", ("vstore", addr_val_b, v_val_b)))

            # Scalar tail for non-multiple of VLEN batch sizes.
            for i in range(vec_batch, batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                # val = mem[inp_values_p + i]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                body.append(("load", ("load", tmp_val, tmp_addr)))
                # node_val = mem[forest_values_p + idx]
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx))
                )
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("&", tmp1, tmp_val, one_const)))
                body.append(("flow", ("select", tmp3, tmp1, two_const, one_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                # mem[inp_indices_p + i] = idx
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const))
                )
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(
                    ("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const))
                )
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
