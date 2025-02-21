# NRSS Major Upgrade: Python GPU Simulation Engine

## Overview
This upgrade will create pure Python GPU-accelerated simulation engines within NRSS as alternatives to the C++ CyRSoXS library. The key goals are:
1. Replicate full CyRSoXS functionality in Python using CuPy and PyTorch backends
2. Maintain compatibility with existing NRSS Material and Morphology classes
3. Enable direct GPU computation without CPU-GPU transfers
4. Improve overall simulation performance

## Phase 1: Core Infrastructure (2-3 weeks)

### 1.1 GPU-Aware Base Classes

#### 1.1.1 Type Management
```python
from enum import Enum, auto
from typing import Union, Dict, Any, Optional
import numpy as np
import cupy as cp

class BackendType(Enum):
    CYRSOXS = auto()  # Original CyRSoXS backend
    CUPY = auto()     # CuPy GPU backend
    TORCH = auto()    # PyTorch GPU backend

class ArraySpec:
    """Specification for array type and precision requirements"""
    def __init__(self, dtype: np.dtype, shape: tuple, required_backend: Optional[BackendType] = None):
        self.dtype = dtype
        self.shape = shape
        self.required_backend = required_backend

class TypeValidator:
    """Validates and manages array types across backends"""
    BACKEND_DTYPES = {
        BackendType.CYRSOXS: {
            'float32': np.float32,
            'complex64': np.complex64
        },
        BackendType.CUPY: {
            'float32': cp.float32,
            'complex64': cp.complex64
        }
    }

    @staticmethod
    def validate_array(array: Any, spec: ArraySpec) -> bool:
        """Validate array matches specification"""
        pass

    @staticmethod
    def get_optimal_type(array: Any, target_backend: BackendType) -> np.dtype:
        """Determine optimal type for array in target backend"""
        pass
```

#### 1.1.2 Array Container
```python
class GPUArrayContainer:
    """Base class for GPU-aware array management with type preservation"""
    def __init__(self, backend: BackendType = BackendType.CYRSOXS):
        self.backend = backend
        self._arrays: Dict[str, Any] = {}  # Lazy-loaded arrays
        self._specs: Dict[str, ArraySpec] = {}  # Type specifications
        self._validator = TypeValidator()

    def register_array(self, name: str, spec: ArraySpec) -> None:
        """Register array specification for type checking"""
        self._specs[name] = spec

    def set_array(self, name: str, array: Any) -> None:
        """Set array with type validation
        
        - Preserves existing np.float32 arrays for CyRSoXS compatibility
        - Allows direct use of cp.float32 arrays for GPU backends
        - Validates type compatibility with backend
        """
        if name not in self._specs:
            raise ValueError(f"Array {name} not registered")

        spec = self._specs[name]
        if not self._validator.validate_array(array, spec):
            if self.backend == BackendType.CYRSOXS:
                # Ensure CyRSoXS gets numpy arrays
                array = self._to_numpy(array, spec.dtype)
            else:
                # For GPU backends, convert only if necessary
                optimal_type = self._validator.get_optimal_type(array, self.backend)
                if array.dtype != optimal_type:
                    array = self._convert_type(array, optimal_type)

        self._arrays[name] = array

    def get_array(self, name: str) -> Any:
        """Get array in the correct type for current backend"""
        if name not in self._arrays:
            raise KeyError(f"Array {name} not found")

        array = self._arrays[name]
        if self.backend == BackendType.CYRSOXS and not isinstance(array, np.ndarray):
            return self._to_numpy(array, self._specs[name].dtype)
        return array

    def _to_numpy(self, array: Any, dtype: np.dtype) -> np.ndarray:
        """Convert array to numpy with minimal copying"""
        if isinstance(array, np.ndarray) and array.dtype == dtype:
            return array
        if hasattr(array, 'get'):
            return array.get().astype(dtype)
        return np.asarray(array, dtype=dtype)

    def _convert_type(self, array: Any, dtype: Any) -> Any:
        """Convert array type with minimal copying"""
        if array.dtype == dtype:
            return array
        return array.astype(dtype)
```

### 1.2 Extended Material/Morphology Classes
- Extend existing Material class to inherit from GPUArrayContainer
- Add GPU-specific properties for optical constants
- Implement lazy loading for large arrays
- Add backend selection (CuPy/PyTorch)

### 1.3 Common Interface
```python
class GPUSimulationEngine(ABC):
    @abstractmethod
    def initialize(self, material: Material, morphology: Morphology) -> None:
        """Initialize simulation with given material and morphology"""
        pass
    
    @abstractmethod
    def compute_scattering(self, energy_points: np.ndarray) -> np.ndarray:
        """Compute scattering pattern for given energy points"""
        pass
```

## Phase 2: CuPy Implementation (4-5 weeks)

### 2.1 Core Components

#### 2.1.1 Data Flow Analysis
Tracing the path from NRSS through CyRSoXS:

1. NRSS Entry Point (`/homes/deand/dev/NRSS/src/NRSS/morphology.py`):
   - Material class holds voxel-level data and optical constants
   - Morphology class manages multiple materials and simulation config
   - Data is passed to CyRSoXS via pybind11 interfaces

2. CyRSoXS Data Ingestion:
   - `VoxelData.h`: Handles morphology data transfer
   - `RefractiveIndex.h`: Manages optical constants
   - `InputData.h`: Processes simulation parameters

3. GPU Computation Flow:
   - `cudaMain.h`: Orchestrates GPU operations
   - `cudaUtils.h`: Provides core GPU utilities

4. Results Return:
   - `ScatteringPattern.h`: Manages computed patterns
   - Data returned to NRSS via numpy arrays

#### 2.1.2 Core Components Implementation

1. Material Data Management:
```python
class CuPyMaterial:
    def __init__(self, backend='cupy'):
        self.backend = backend
        self._gpu_arrays = {}
        
    def load_material_data(self, material: Material):
        """Load NRSS Material data to GPU"""
        # Reference: cyrsoxs/include/PyClass/VoxelData.h
        pass
        
    def load_optical_constants(self, material: Material):
        """Load optical constants to GPU"""
        # Reference: cyrsoxs/include/PyClass/RefractiveIndex.h
        pass
```

2. FFT Operations:
```python
class CuPyFFTEngine:
    def __init__(self):
        self.fft_cache = {}
        
    def perform_fft(self, polarization):
        """3D FFT implementation
        Reference: cyrsoxs/include/cudaMain.h:performFFT"""
        pass
        
    def perform_fft_shift(self, data, dims):
        """FFT shift operation
        Reference: cyrsoxs/include/cudaMain.h:performFFTShift"""
        pass
```

3. Polarization Computation:
```python
class PolarizationEngine:
    def compute_polarization(self, material_constants, voxel_data):
        """Compute polarization vectors
        Reference: cyrsoxs/include/cudaMain.h:computePolarization"""
        pass
        
    def compute_nt(self, material_constants, voxel_data):
        """Compute Nt tensor
        Reference: cyrsoxs/include/cudaMain.h:computeNt"""
        pass
```

4. Scattering Computation:
```python
class ScatteringEngine:
    def compute_scatter_3d(self, polarization_xyz, k_vector):
        """Compute 3D scattering pattern
        Reference: cyrsoxs/include/cudaMain.h:performScatter3DComputation"""
        pass
        
    def perform_ewald_projection(self, scatter_data, k_magnitude):
        """Project onto Ewald sphere
        Reference: cyrsoxs/include/cudaMain.h:peformEwaldProjectionGPU"""
        pass
```

5. Complex Matrix Operations:
```python
class ComplexMatrixOps:
    def compute_3x3_complex_multiplication(self, matrix):
        """Complex matrix multiplication
        Reference: cyrsoxs/include/cudaUtils.h:compute3X3ComplexMultiplication"""
        pass
        
    def compute_matrix_vector(self, matrix, vector):
        """Matrix-vector operations
        Reference: cyrsoxs/include/cudaUtils.h:computeMatrixTimesVector"""
        pass
```

#### 2.1.3 Memory Management

1. GPU Memory Allocation:
```python
class GPUMemoryManager:
    def __init__(self):
        self.memory_pool = None  # CuPy memory pool
        
    def allocate_for_computation(self, voxel_dims, num_materials):
        """Smart memory allocation based on problem size"""
        pass
        
    def release_computation_memory(self):
        """Clean up GPU memory"""
        pass
```

2. Stream Management:
```python
class StreamManager:
    def __init__(self):
        self.streams = {}
        
    def get_stream(self, operation_type):
        """Get appropriate CUDA stream for operation"""
        pass
```

### 2.2 Memory Management
1. Implement smart caching:
   - Cache frequently used arrays
   - Use pinned memory for faster transfers
   - Implement memory pool for intermediate results

2. Stream Processing:
   - Add CUDA stream support via CuPy
   - Enable concurrent kernel execution
   - Implement batch processing

## Phase 3: PyTorch Implementation (4-5 weeks)

### 3.1 Core Implementation
1. Tensor Operations:
   - Implement complex tensor operations
   - Port matrix calculations to PyTorch
   - Use PyTorch's FFT modules

2. Automatic Differentiation:
   - Leverage PyTorch's autograd
   - Enable gradient computation for future extensions
   - Implement custom backward passes

### 3.2 Optimization Features
1. JIT Compilation:
   - Use TorchScript for core computations
   - Implement kernel fusion
   - Optimize memory access patterns

2. Multi-GPU Support:
   - Add DistributedDataParallel support
   - Implement model parallelism
   - Add GPU selection logic

## Phase 4: Integration and Validation (3-4 weeks)

### 4.1 Engine Selection System
```python
class EngineSelector:
    def __init__(self):
        self.available_backends = self._detect_backends()
        
    def get_optimal_engine(self, problem_size, memory_requirement):
        """Select best engine based on hardware and problem"""
        pass

    def fallback_strategy(self):
        """Define fallback behavior if preferred engine fails"""
        pass
```

### 4.2 Validation Suite
1. Numerical Validation:
   - Compare against CyRSoXS results
   - Validate across different GPU architectures
   - Test numerical stability

2. Performance Benchmarks:
   - Memory transfer benchmarks
   - Computation time comparisons
   - Memory usage analysis

## Phase 5: Documentation and Testing (2-3 weeks)

### 5.1 Documentation
1. API Documentation:
   - Document new GPU interfaces
   - Add migration guides
   - Include performance optimization tips

2. Examples:
   - Add Jupyter notebooks
   - Create benchmark scripts
   - Document common workflows

### 5.2 Testing Infrastructure
1. Unit Tests:
   - Test GPU operations
   - Validate numerical accuracy
   - Test memory management

2. Integration Tests:
   - End-to-end workflow tests
   - Cross-backend validation
   - Performance regression tests

## Success Criteria
1. Functionality:
   - 100% feature parity with CyRSoXS
   - Numerical accuracy within 1e-6
   - Successful validation with experimental data

2. Performance:
   - Equal or better performance than CyRSoXS
   - 90% reduction in CPU-GPU transfers
   - Efficient memory usage (<80% peak GPU memory)

## Timeline
Total Duration: 15-20 weeks
- Phase 1: 2-3 weeks
- Phase 2: 4-5 weeks
- Phase 3: 4-5 weeks
- Phase 4: 3-4 weeks
- Phase 5: 2-3 weeks

## Dependencies
- CuPy >= 12.0.0
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Python >= 3.8
- CUDA Toolkit >= 11.0

## Risk Mitigation
1. Technical Risks:
   - Memory limitations
   - Numerical precision differences
   - Performance bottlenecks

2. Mitigation Strategies:
   - Implement progressive loading
   - Add extensive validation
   - Profile-driven optimization

## Future Extensions
1. Potential Features:
   - Multi-GPU processing
   - Distributed computing
   - Additional backends (JAX)
