# Elaston

Elaston is a simple and lightweight module for linear elasticity calculations.

## Installation

We will offer pip and conda at some point. For now, you can install the package by cloning the repository and running the following command in the root directory:

```bash
pip install .
```

## Features

- Calculation of elastic constants
- Stress strain field around dislocations using anisotropic elasticity
- Stress strain field around point defects


## Usage

Examples I: Get bulk modulus from the elastic tensor:

```python
from elaston import LinearElasticity

medium = LinearElasticity(elastic_tensor)
print(medium.bulk_modulus)
```


Example II: Get strain field around a point defect:

```python
import numpy as np
medium = LinearElasticity(elastic_tensor)
random_positions = np.random.random((10, 3))-0.5
dipole_tensor = np.eye(3)
print(medium.get_point_defect_strain(random_positions, dipole_tensor))
```


Example III: Get stress field around a dislocation:

```python
import numpy as np
medium = LinearElasticity(elastic_tensor)
random_positions = np.random.random((10, 3))-0.5
burgers_vector = np.array([0, 0, 1])
print(medium.get_dislocation_stress(random_positions, burgers_vector))
```

Example IV: Estimate the distance between partial dislocations:

```python
medium = LinearElasticity(elastic_tensor)
partial_one = np.array([-0.5, 0, np.sqrt(3)/2])*lattice_constant
partial_two = np.array([0.5, 0, np.sqrt(3)/2])*lattice_constant
distance = 100
stress_one = medium.get_dislocation_stress([0, distance, 0], partial_one)
print('Choose `distance` in the way that the value below corresponds to SFE')
medium.get_dislocation_force(stress_one, [0, 1, 0], partial_two)
```
