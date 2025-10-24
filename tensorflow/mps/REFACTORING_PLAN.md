# MPS Backend Refactoring Plan

## Objectif
Restructurer le backend MPS monolithique (6000+ lignes) en une architecture modulaire professionnelle.

## Structure cible

```
tensorflow/mps/
├── BUILD                           # Bazel build config
├── README.md                       # Documentation du backend
│
├── core/                           # Core device infrastructure
│   ├── mps_device.h               # Device interface
│   ├── mps_device.mm              # Device implementation
│   ├── mps_stream.h               # Stream management
│   ├── mps_stream.mm              # Stream implementation  
│   ├── mps_event.h                # Event synchronization
│   ├── mps_event.mm               # Event implementation
│   └── mps_allocator.mm           # Memory allocator
│
├── kernels/                        # Kernel implementations
│   ├── mps_elementwise_ops.mm     # Unary/Binary ops (Add, Mul, Relu, etc.)
│   ├── mps_comparison_ops.mm      # Comparison ops (Equal, Less, etc.)
│   ├── mps_logical_ops.mm         # Logical ops (And, Or, Not)
│   ├── mps_reduction_ops.mm       # Reductions (Sum, Mean, All, Any)
│   ├── mps_tensor_ops.mm          # Tensor ops (Reshape, Transpose, etc.)
│   ├── mps_slice_ops.mm           # Slice, StridedSlice
│   ├── mps_gather_scatter_ops.mm  # Gather, Scatter variants
│   ├── mps_split_concat_ops.mm    # Split, Concat
│   ├── mps_nn_ops.mm              # Conv2D, MatMul, Pooling
│   ├── mps_activation_ops.mm      # Activations (Gelu, Swish, etc.)
│   ├── mps_normalization_ops.mm   # BatchNorm, etc.
│   └── mps_utility_ops.mm         # Fill, Range, OneHot, etc.
│
├── ops/                            # Registration layer
│   ├── mps_ops_registry.cc        # Main registration
│   ├── mps_ops_macros.h           # Registration macros
│   └── mps_kernel_helpers.h       # Common kernel utilities
│
├── utils/                          # Utilities
│   ├── mps_dtype_utils.h          # Dtype conversion helpers
│   ├── mps_shape_utils.h          # Shape manipulation
│   ├── mps_broadcast_utils.h      # Broadcasting logic
│   └── mps_graph_helpers.h        # MPSGraph common patterns
│
└── tests/                          # Tests
    ├── ops_test.py                # Python tests
    ├── benchmark_ops.py           # Benchmarks
    └── smoke_test.py              # Integration tests
```

## Migration par phases

### Phase 1: Extraction des headers (Semaine 1)
- [ ] Créer les fichiers .h avec les déclarations
- [ ] Extraire les structures communes (MPSDevice, MPSStream, etc.)
- [ ] Créer mps_ops_macros.h avec toutes les macros

### Phase 2: Séparation des kernels par catégorie (Semaine 2)
- [ ] mps_elementwise_ops.mm (41 ops unary/binary)
- [ ] mps_comparison_ops.mm (6 ops: Equal, NotEqual, etc.)
- [ ] mps_logical_ops.mm (3 ops: And, Or, Not)
- [ ] mps_reduction_ops.mm (7 ops: Sum, Mean, All, Any, etc.)

### Phase 3: Tensor operations (Semaine 3)
- [ ] mps_tensor_ops.mm (Cast, Reshape, Transpose, Concat)
- [ ] mps_slice_ops.mm (Slice, StridedSlice)
- [ ] mps_gather_scatter_ops.mm (Gather*, Scatter*)
- [ ] mps_split_concat_ops.mm (Split, Concat)

### Phase 4: NN operations (Semaine 4)
- [ ] mps_nn_ops.mm (Conv2D, DepthwiseConv2D, MatMul, Pooling)
- [ ] mps_activation_ops.mm (Activations)
- [ ] mps_normalization_ops.mm (BatchNorm)

### Phase 5: Infrastructure finale (Semaine 5)
- [ ] mps_device.mm (Device/Stream/Event/Allocator)
- [ ] mps_ops_registry.cc (Registrations centralisées)
- [ ] Créer BUILD avec tous les targets
- [ ] Mise à jour de la documentation

## Bénéfices attendus

### Compilaton
- ✅ Builds incrémentaux plus rapides
- ✅ Parallélisation de la compilation
- ✅ Temps de rebuild réduit de ~70%

### Maintenabilité
- ✅ Code organisé par fonctionnalité
- ✅ Tests unitaires par catégorie
- ✅ Debugging facilité
- ✅ Onboarding plus simple

### Scalabilité
- ✅ Ajout de nouveaux ops simplifié
- ✅ Tests isolés par module
- ✅ Réutilisation de code (utils/)

## Compatibilité

- ✅ Pas de changement d'API publique
- ✅ Registration identique
- ✅ Backward compatible à 100%
- ✅ Tests existants inchangés

## Timeline

**Durée totale**: 5 semaines (part-time) ou 2 semaines (full-time)

**Milestone 1** (après Phase 2): Elementwise/Comparison/Logical ops modularisés
**Milestone 2** (après Phase 3): Tensor ops modularisés  
**Milestone 3** (après Phase 5): Refactoring complet

## Notes

- Garder le fichier monolithique jusqu'à ce que tous les modules soient testés
- Migration progressive: nouveau code va dans les modules, ancien reste
- Supprimer le monolithe seulement quand 100% migré et testé
