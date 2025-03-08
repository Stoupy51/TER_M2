
## 1. Setting Up Data and Preprocessing
# Imports
import stouputils as stp
import pennylane as qml
import numpy as np
from itertools import combinations
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from qiskit import QuantumCircuit#, IBMQ
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
# from qiskit.providers.ibmq import least_busy
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import TrainableKernel as QuantumKernel

# Set seed
SEED: int = 1234
np.random.seed(SEED)

def load_and_preprocess_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Load and preprocess the wine dataset.
    
    Loads the wine dataset, splits it into training and test sets,
    and scales the data to the range [0, 1].
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Preprocessed training and test data and labels.
    """
    # Data Loading and Splitting: Loads the wine dataset and splits it into training and test sets,
    # using 90% of the data for training and 10% for testing.
    x, y = load_wine(return_X_y=True)
    x = x[:59+71]
    y = y[:59+71]
    x_tr, x_test, y_tr, y_test = train_test_split(x, y, train_size=0.9)

    # Scaling: Uses MaxAbsScaler to scale the data to the range [0, 1], which is suitable for quantum computing tasks.
    scaler: MaxAbsScaler = MaxAbsScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_test = scaler.transform(x_test)
    x_test = np.clip(x_test, 0, 1)	# Clip dataset to be between 0 and 1    # type: ignore
    
    return x_tr, x_test, y_tr, y_test

x_tr, x_test, y_tr, y_test = load_and_preprocess_data()
stp.whatisit(x_tr, x_test, y_tr, y_test, max_length=25)


## 2. Quantum Kernel and Support Vector Machine
def amplitude_embedding_qsvm() -> None:
    """ Implements Quantum SVM using amplitude embedding.
    
    Defines a quantum kernel circuit using Pennylane with amplitude embedding,
    and trains an SVM classifier using this kernel.
    
    Returns:
        None: This function prints the accuracy score.
    """
    # Quantum Circuit: Defines a quantum kernel circuit using Pennylane. This circuit uses amplitude embedding and measures the probabilities.
    nqubits: int = 4
    dev = qml.device("lightning.qubit", wires=nqubits)

    @qml.qnode(dev)
    def kernel_circ(a, b):
        """ Quantum circuit for kernel computation.
        
        Embeds data points into quantum states and computes their overlap.
        
        Args:
            a (np.ndarray): First data point to embed.
            b (np.ndarray): Second data point to embed.
            
        Returns:
            np.ndarray: Probability distribution of the quantum state.
        """
        qml.AmplitudeEmbedding(
            a, wires=range(nqubits), pad_with=0, normalize=True)
        qml.adjoint(qml.AmplitudeEmbedding(
            b, wires=range(nqubits), pad_with=0, normalize=True))
        return qml.probs(wires=range(nqubits))

    # Quantum Kernel Function: Implements the quantum kernel function, which computes the kernel matrix for the Support Vector Machine (SVM) using the quantum circuit.
    def qkernel(A, B):
        """ Quantum kernel function.
        
        Computes the kernel matrix between two sets of data points.
        
        Args:
            A (np.ndarray): First set of data points.
            B (np.ndarray): Second set of data points.
            
        Returns:
            np.ndarray: Kernel matrix.
        """
        return np.array([[kernel_circ(a, b)[0] for b in B] for a in A])

    # SVM Training and Evaluation: Trains an SVM classifier using the quantum kernel and evaluates its performance on the test set.
    svm = SVC(kernel=qkernel).fit(x_tr, y_tr)

    stp.info(f"Amplitude Embedding QSVM Accuracy: {accuracy_score(svm.predict(x_test), y_test)}")

amplitude_embedding_qsvm()


## 3. PCA for Dimensionality Reduction
def pca_angle_embedding_qsvm() -> None:
    """ Implements Quantum SVM with PCA and angle embedding.
    
    Reduces data dimensionality using PCA, then defines a quantum kernel
    with angle embedding and trains an SVM classifier.
    
    Returns:
        None: This function prints the accuracy score.
    """
    # PCA Transformation: Reduces the dimensionality of the data to 8 components using PCA.
    pca = PCA(n_components=8)
    xs_tr = pca.fit_transform(x_tr)
    xs_test = pca.transform(x_test)

    # Quantum Circuit: Redefines the quantum kernel circuit with angle embedding instead of amplitude embedding.
    nqubits: int = 8
    dev = qml.device("lightning.qubit", wires=nqubits)

    @qml.qnode(dev)
    def kernel_circ(a, b):
        """ Quantum circuit for kernel computation using angle embedding.
        
        Embeds data points into quantum states using angle embedding and computes their overlap.
        
        Args:
            a (np.ndarray): First data point to embed.
            b (np.ndarray): Second data point to embed.
            
        Returns:
            np.ndarray: Probability distribution of the quantum state.
        """
        qml.AngleEmbedding(a, wires=range(nqubits))
        qml.adjoint(qml.AngleEmbedding(b, wires=range(nqubits)))
        return qml.probs(wires=range(nqubits))

    # Quantum Kernel Function
    def qkernel(A, B):
        """ Quantum kernel function.
        
        Computes the kernel matrix between two sets of data points.
        
        Args:
            A (np.ndarray): First set of data points.
            B (np.ndarray): Second set of data points.
            
        Returns:
            np.ndarray: Kernel matrix.
        """
        return np.array([[kernel_circ(a, b)[0] for b in B] for a in A])

    # SVM Training and Evaluation: Trains an SVM classifier with the new quantum kernel and evaluates its performance.
    svm = SVC(kernel=qkernel).fit(xs_tr, y_tr)

    stp.info(f"PCA Angle Embedding QSVM Accuracy: {accuracy_score(svm.predict(xs_test), y_test)}")

pca_angle_embedding_qsvm()


## 4. ZZFeatureMap and SVM with Quantum Kernel
def zz_feature_map_qsvm() -> None:
    """ Implements Quantum SVM with ZZFeatureMap.
    
    Defines a custom ZZ feature map for embedding data into quantum states,
    and trains an SVM classifier using this feature map.
    
    Returns:
        None: This function prints the accuracy score.
    """
    def ZZFeatureMap_custom(nqubits: int, data: np.ndarray) -> None:
        """ ZZ Feature Map for quantum kernel.
        
        Defines a feature map based on ZZ interactions and RZ rotations to embed
        classical data into quantum states.
        
        Args:
            nqubits (int): Number of qubits to use.
            data (np.ndarray): Data point to embed.
        """
        nload = min(len(data), nqubits)
        for i in range(nload):
            qml.Hadamard(i)
            qml.RZ(2.0 * data[i], wires=i)
        for pair in list(combinations(range(nload), 2)):
            q0 = pair[0]
            q1 = pair[1]
            qml.CZ(wires=[q0, q1])
            qml.RZ(2.0 * (np.pi - data[q0]) * (np.pi - data[q1]), wires=q1)
            qml.CZ(wires=[q0, q1])

    # PCA Transformation
    pca = PCA(n_components=8)
    xs_tr = pca.fit_transform(x_tr)
    xs_test = pca.transform(x_test)

    # Quantum Circuit: Uses the ZZFeatureMap function in the quantum circuit
    nqubits = 4
    dev = qml.device("lightning.qubit", wires=nqubits)

    @qml.qnode(dev)
    def kernel_circ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """ Quantum circuit for kernel computation using ZZFeatureMap.
        
        Embeds data points into quantum states using ZZFeatureMap and computes their overlap.
        
        Args:
            a (np.ndarray): First data point to embed.
            b (np.ndarray): Second data point to embed.
            
        Returns:
            np.ndarray: Probability distribution of the quantum state.
        """
        ZZFeatureMap_custom(nqubits, a)
        qml.adjoint(ZZFeatureMap_custom)(nqubits, b)
        return qml.probs(wires=range(nqubits))

    # Quantum Kernel Function
    def qkernel(A, B):
        """ Quantum kernel function.
        
        Computes the kernel matrix between two sets of data points.
        
        Args:
            A (np.ndarray): First set of data points.
            B (np.ndarray): Second set of data points.
            
        Returns:
            np.ndarray: Kernel matrix.
        """
        return np.array([[kernel_circ(a, b)[0] for b in B] for a in A])

    # SVM Training and Evaluation: Trains an SVM classifier with this new quantum kernel
    svm = SVC(kernel=qkernel).fit(xs_tr, y_tr)
    stp.info(f"ZZ Feature Map QSVM Accuracy: {accuracy_score(svm.predict(xs_test), y_test)}")

zz_feature_map_qsvm()


## 6. Using Qiskit for Quantum SVM
def qiskit_qsvm() -> None:
    """ Implements Quantum SVM using Qiskit.
    
    Demonstrates how to use Qiskit for creating parameterized quantum circuits,
    implementing ZZFeatureMap, and training a Quantum SVM classifier.
    
    Returns:
        None: This function prints the accuracy score.
    """
    # PCA Transformation
    pca: PCA = PCA(n_components=8)
    xs_tr: np.ndarray = pca.fit_transform(x_tr)
    xs_test: np.ndarray = pca.transform(x_test)

    # Quantum Circuit with Parameters
    # Example 1: Single parameter circuit
    parameter: Parameter = Parameter("x")
    qc1: QuantumCircuit = QuantumCircuit(1)
    qc1.rx(parameter, 0)
    
    # Example 2: Parameter vector circuit
    parameter_vector: ParameterVector = ParameterVector("x", length=2)
    qc2: QuantumCircuit = QuantumCircuit(2)
    qc2.rx(parameter_vector[0], 0)
    qc2.rx(parameter_vector[1], 1)
    
    # ZZFeatureMap and QuantumKernel
    zzfm: ZZFeatureMap = ZZFeatureMap(8)
    qkernel: QuantumKernel = QuantumKernel(feature_map=zzfm, quantum_instance=AerSimulator())
    
    # QSVC: Quantum Support Vector Classification
    qsvm: QSVC = QSVC(quantum_kernel=qkernel)
    qsvm.fit(xs_tr, y_tr)
    
    stp.info(f"Qiskit QSVM Accuracy: {accuracy_score(qsvm.predict(xs_test), y_test)}")

qiskit_qsvm()


# ## 7. Using IBM Quantum Devices
# def ibm_quantum_qsvm(ibm_token: str = "1234") -> None:
#     """ Implements Quantum SVM using IBM Quantum hardware.
    
#     Sets up an IBM Quantum account, selects an appropriate quantum device,
#     and trains a Quantum SVM with dimensionality reduction.
    
#     Note: To use this function, replace "1234" with your actual IBM token
    
#     Args:
#         ibm_token (str): IBM Quantum authentication token.
        
#     Returns:
#         None: This function trains a model on IBM quantum hardware.
#     """
#     # IBM Quantum Account setup
#     IBMQ.save_account(ibm_token)
#     provider = IBMQ.load_account()
    
#     # Device Selection
#     dev_list = provider.backends(
#         filters=lambda x: x.configuration().n_qubits >= 7,
#         simulator=False
#     )
#     dev = least_busy(dev_list)
    
#     # PCA and Quantum SVM
#     # Reduce dimensionality to 7 components
#     pca: PCA = PCA(n_components=7)
#     xss_tr: np.ndarray = pca.fit_transform(x_tr)
#     xss_test: np.ndarray = pca.transform(x_test)
    
#     # Create feature map and quantum kernel
#     zzfm: ZZFeatureMap = ZZFeatureMap(7)
#     qkernel: QuantumKernel = QuantumKernel(feature_map=zzfm, quantum_instance=dev)
    
#     # Train QSVC model
#     qsvm: QSVC = QSVC(quantum_kernel=qkernel)
#     qsvm.fit(xss_tr, y_tr)
    
#     stp.info(f"IBM Quantum QSVM Accuracy: {accuracy_score(qsvm.predict(xss_test), y_test)}")

# ibm_quantum_qsvm()
