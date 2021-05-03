from qiskit import QuantumCircuit, execute, Aer, IBMQ , ClassicalRegister,QuantumRegister,execute
from qiskit.compiler import transpile, assemble
#from qiskit.tools.jupyter import *
from qiskit.visualization import *
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import networkx as nx
from fractions import Fraction

pi = np.pi

# Functions
def ccr(alpha,theta,c1,c2,u,qc,n): # control-control-rotation gate gate
    
    pi = np.pi
    sim = QuantumRegister(n,'sim')
    # alpha is x,y,z. u is qubit that u acts on. c_1,c_2 are the control locations
    if alpha == 'x':
        qc.cu3(theta/2,-pi/2,pi/2,sim[c2],sim[u])
        qc.cx(sim[c1],sim[c2])
        qc.cu3(-theta/2,-pi/2,pi/2,sim[c2],sim[u])
        qc.cx(sim[c1],sim[c2])
        qc.cu3(theta/2,-pi/2,pi/2,sim[c1],sim[u])
    elif alpha == 'y':
        qc.cu3(theta/2,0,0,sim[c2],sim[u])
        qc.cx(sim[c1],sim[c2])
        qc.cu3(-theta/2,0,0,sim[c2],sim[u])
        qc.cx(sim[c1],sim[c2])
        qc.cu3(theta/2,0,0,sim[c1],sim[u])
    else:
        qc.cu1(theta/2,sim[c2],sim[u])
        qc.cx(sim[c1],sim[c2])
        qc.cu1(-theta/2,sim[c2],sim[u])
        qc.cx(sim[c1],sim[c2])
        qc.cu1(theta/2,sim[c1],sim[u])
    
def scs(x,y,qc,n): # s is starting qubit, qc is quantum circuit
    sim = QuantumRegister(n,'sim')
    for i in range(1,y+1):
        if i == 1:
            qc.cx(sim[x-1-i],sim[x-1])
            qc.cu3(2*math.acos(math.sqrt(i/x)),0,0,sim[x-1],sim[x-1-i])
            qc.cx(sim[x-1-i],sim[x-1])
        else:
            qc.cx(sim[x-1-i],sim[x-1])
            ccr('x',2*math.acos(math.sqrt(i/x)),x-1,x-i,x-1-i,qc,n)
            qc.cx(sim[x-1-i],sim[x-1])

# Returns Dicke circuit
def Dicke_exp(n,k):
    
    # Create circuit
    sim = QuantumRegister(n,'sim')
    meas = ClassicalRegister(n,'meas')
    circ = QuantumCircuit(sim, meas)
    
    # Prepare initial bit string
    for i in range(n-1,n-k-1,-1):
        circ.x(sim[i])
    
    for l in range(n,k,-1):
        scs(l,k,circ,n)
    for l in range(k,1,-1):
        scs(l,l-1,circ,n)
        
    # Measure
    #circ.measure(sim,meas)
    
    trans = transpile(circ, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=3)

    # Draw
    return circ

def append_zz_term(qc,q1,q2,gamma,N):
    sim = QuantumRegister(N,'sim')
    qc.cx(sim[q1],sim[q2])
    qc.rz(2*gamma,sim[q2])
    qc.cx(sim[q1],sim[q2])
    
def append_z_term(qc,q1,gamma,N):
    sim = QuantumRegister(N,'sim')
    qc.rz(2*gamma,sim[q1])

def get_cost_operator_circuit(G,gamma):
    N = G.number_of_nodes()
    #qc = QuantumCircuit(N,N)
    sim = QuantumRegister(N,'sim')
    meas = ClassicalRegister(N,'meas')
    qc = QuantumCircuit(sim, meas)
    for i,j in G.edges():
        append_zz_term(qc,i,j,gamma,N)
        append_z_term(qc,i,gamma,N)
        append_z_term(qc,j,gamma,N)
    return qc


def append_x_term(qc,q1,beta,N):
    sim = QuantumRegister(N,'sim')
    qc.rx(2*beta,sim[q1])
    
def swap_mixer(qc,q1,q2,gamma,N):
    sim = QuantumRegister(N,'sim')
    qc.rxx(2*gamma,sim[q1],sim[q2])
    qc.ryy(2*gamma,sim[q1],sim[q2])


def get_ring_mixer_operator_circuit(G,beta):
    N = G.number_of_nodes()
    sim = QuantumRegister(N,'sim')
    meas = ClassicalRegister(N,'meas')
    qc = QuantumCircuit(sim, meas)
    for i in G.nodes():
        swap_mixer(qc,i,(i+1)%N,beta,N)
    return qc

def get_complete_mixer_operator_circuit(G,beta):
    N = G.number_of_nodes()
    sim = QuantumRegister(N,'sim')
    meas = ClassicalRegister(N,'meas')
    qc = QuantumCircuit(sim, meas)
    for i,j in G.edges():
        swap_mixer(qc,i,j,beta,N)
    return qc

def get_qaoa_circuit(G,beta,gamma,n,k,mixer):
    assert(len(beta) == len(gamma))
    p = len(beta)
    N = G.number_of_nodes()
    qc = Dicke_exp(n,k)
    #qc = QuantumCircuit(N,N)
    #qc.h(range(N))
    if mixer == 1:
        for i in range(p):
           qc += get_cost_operator_circuit(G,gamma[i])
           qc += get_complete_mixer_operator_circuit(G,beta[i])
    elif mixer == 2:
           qc += get_cost_operator_circuit(G,gamma[i])
           qc += get_ring_mixer_operator_circuit(G,beta[i])
    qc.barrier(range(N))
    qc.measure(range(N),range(N))
    return qc

def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}

def maxkvertex_obj(x,G):
    cut = 0
    for i,j in G.edges():
        if x[i] == '1' or x[j] == '1':
            cut -= 1
    return cut

def compute_maxcut_energy(counts,G):
    energy = 0
    total_counts = 0
    for meas,meas_count in counts.items():
        obj_for_meas = maxkvertex_obj(meas,G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy/total_counts

def get_black_box_objective(G,p,n,k,mixer):
    backend = Aer.get_backend('qasm_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc = get_qaoa_circuit(G,beta,gamma,n,k,mixer)
        counts = execute(qc,backend,seed_simulator=10).result().get_counts()
        return compute_maxcut_energy(invert_counts(counts),G)
    return f

###############################

def get_qaoa_circuit_sv(G,beta,gamma,n,k,mixer):
    assert(len(beta) == len(gamma))
    p = len(beta)
    N = G.number_of_nodes()
    qc = Dicke_exp(n,k)
    #qc = QuantumCircuit(N,N)
    #qc.h(range(N))
    if mixer == 1:
        for i in range(p):
           qc += get_cost_operator_circuit(G,gamma[i])
           qc += get_complete_mixer_operator_circuit(G,beta[i])
    elif mixer == 2:
        for i in range(p):
           qc += get_cost_operator_circuit(G,gamma[i])
           qc += get_ring_mixer_operator_circuit(G,beta[i])
    #qc.measure(range(N),range(N))
    return qc

def state_to_ampl_counts(vec, eps=1e-15):
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Not Valis")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = '0{}b'.format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2+val.imag**2 > eps:
            counts[format(kk,str_format)] = val
    return counts

def state_num2str(basis_state_as_num, nqubits):
    return '{0:b}'.format(basis_state_as_num).zfill(nqubits)

def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)

def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)

def get_adjusted_state(state):
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector not valid")
    nqubits = int(nqubits)
    
    adjusted_state = np.zeros(2**nqubits, dtype=complex)
    for basis_state in range(2**nqubits):
        adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state

def compute_maxkvertex_energy_sv(sv,G):
    counts = state_to_ampl_counts(sv)
    
    return sum(maxkvertex_obj(np.array([x for x in l]), G) * (np.abs(v)**2) for l, v in counts.items())

def get_black_box_objective_sv(G,p,n,k,mixer):
    backend = Aer.get_backend('statevector_simulator')
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        cc = get_qaoa_circuit_sv(G,beta,gamma,n,k,mixer)
        svv = get_adjusted_state(execute(cc, backend).result().get_statevector())
        return compute_maxkvertex_energy_sv(svv,G)
    return f
