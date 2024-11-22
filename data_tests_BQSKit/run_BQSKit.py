"""This example shows how to synthesize a circuit with BQSKit."""
from __future__ import annotations
import time
import numpy as np

from ion_gates_BQSKIt import MSGate, XYGate
from bqskit.passes.search import AStarHeuristic, DijkstraHeuristic, GreedyHeuristic
from bqskit.ir.gates import RZGate
from quantum_circuits.quantum_gates_numba import create_standard_ms_set
from quantum_circuits.gate_set_numba import create_fast_ms_set_numba
from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.compiler import CompilationTask
from bqskit.passes import WideLayerGenerator
from bqskit.passes import QSearchSynthesisPass


from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings



def main():
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

    # Construct the unitary as an NumPy array
    toffoli = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]

    sx = np.array([[0, 1], [1, 0]], np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], np.complex128)
    sz = np.array([[1, 0], [0, -1]], np.complex128)
    I = np.identity(2, np.complex128)

    n = 3
    gate_set_fast_nb, gate_names_fast_nb, zprod = create_fast_ms_set_numba(n)
    gate_set_nb, gate_names_nb, gate_grad_nb = create_standard_ms_set(n)

    #ms_fast_nb = gate_set_fast_nb[0]
    #ms_nb = gate_set_nb[0]

    #xy_fast_nb = gate_set_fast_nb[1]

    #ms_fixed = ConstantUnitaryGate(UnitaryMatrix(ms_gate(np.pi / 2, 0)[0]))
    #xy_fixed = ConstantUnitaryGate(UnitaryMatrix(xy_gate(np.pi / 4, 0)[0]))
    # target_gate_set = {ZGate}
    # constant_gate_set = {ms_fixed, RZGate(), RXGate(), RYGate()}
    # antoher_gate_set = {RZGate(), RXXGate(), RYYGate(), RXGate(), RYGate()}
    #parametric_gate_set = {MSGate, RZGate(), RXGate()}

    #model = MachineModel(n, gate_set=parametric_gate_set)

    #compiled_circuit = compile(toffoli, model, with_mapping=False)
    # print("Gate Counts:", compiled_circuit.count(ms_fixed))
    # print("Gate Counts:", compiled_circuit.count(xy_fixed))
    # print("Gate Counts:", compiled_circuit.count(RZGate()))
    # print("Gate Counts:", compiled_circuit.count(RXGate()))

    layer_gen = WideLayerGenerator(multi_qudit_gates={MSGate(n), XYGate(n)}, single_qudit_gate=RZGate())
    # layer_gen = SimpleLayerGenerator(two_qudit_gate=CNOTGate(), single_qudit_gate_1=BGate())

    heuristics = [AStarHeuristic(), DijkstraHeuristic(), GreedyHeuristic()]
    hf_names = ["A*", "Dijkstra", "Greedy"]

    for i_hf, hf in enumerate(heuristics):
        t0 = time.time()
        configured_qsearch_pass = QSearchSynthesisPass(layer_generator=layer_gen, heuristic_function=hf,
                                                       success_threshold=1e-2)

        # Create and execute a compilation task
        toffoli_circuit = Circuit.from_unitary(toffoli)
        # print(ms.get_unitary([0, 0]))
        # print(ms.get_grad([0, 0]))

        with Compiler() as compiler:
            task = CompilationTask(toffoli_circuit, [configured_qsearch_pass])
            synthesized_circuit = compiler.compile(task)

        # print(synthesized_circuit)
        print(synthesized_circuit.gate_set)
        print(hf)
        t1 = time.time() - t0
        print(f"Time: {t1}")

        with open(f"synthesized_circuit_{hf_names[i_hf]}.txt", "w") as f:
            for gate in synthesized_circuit.gate_set:
                print(f"{gate} Count:", synthesized_circuit.count(gate))
                f.write(f"{gate} Count: {synthesized_circuit.count(gate)}\n")
            #f.write(f"Cost: {synthesized_circuit.cost}\n")
            #f.write(f"Depth: {synthesized_circuit.depth}\n")
            f.write(f"Depth: {synthesized_circuit.depth}\n")
            # time
            f.write(f"Time: {t1}\n")



if __name__ == "__main__":
    main()
