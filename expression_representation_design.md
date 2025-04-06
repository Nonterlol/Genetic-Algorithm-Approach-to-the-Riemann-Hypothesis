# Mathematical Expression Representation Design

This document outlines the design for representing mathematical expressions in our genetic algorithm approach to the Riemann Hypothesis.

## Tree-Based Representation

We will use a tree-based representation for mathematical expressions, which is the standard approach in genetic programming and symbolic regression:

### Node Structure

```python
class Node:
    def __init__(self, type, value=None):
        self.type = type  # 'operator', 'function', 'variable', 'constant'
        self.value = value  # The actual operator, function, variable name, or constant value
        self.children = []  # Child nodes
```

### Expression Types

1. **Operators**:
   - Binary: +, -, *, /, ^
   - Unary: - (negation)
   - Complex arithmetic: complex addition, multiplication, etc.

2. **Functions**:
   - Standard mathematical: sin, cos, exp, log
   - Special functions: zeta, gamma
   - Complex functions: abs, arg, re, im

3. **Variables**:
   - Complex variable: s (standard for Riemann zeta function)
   - Real variables: t (imaginary part of s), x, y
   - Integer variables: n (for summations)

4. **Constants**:
   - Mathematical constants: π, e, i
   - Numerical constants: integers, rational numbers, floating-point values
   - Special values: 1/2 (critical line)

## Expression Generation

### Random Expression Generation

For initial population and mutation operations:

1. **Growth Method**:
   - Start with a root node
   - Recursively add child nodes with decreasing probability as depth increases
   - Stop at maximum depth with terminal nodes (variables or constants)

2. **Full Method**:
   - Create full trees of specified depth
   - All branches reach exactly the specified depth

3. **Ramped Half-and-Half**:
   - Combine growth and full methods
   - Create trees of various depths for diversity

### Expression Constraints

To ensure mathematical validity:

1. **Type Checking**:
   - Ensure functions receive appropriate input types
   - Handle complex vs. real values appropriately

2. **Domain Constraints**:
   - Avoid division by zero
   - Ensure logarithms have positive arguments
   - Handle complex domains for special functions

3. **Structural Constraints**:
   - Maximum depth to prevent bloat
   - Maximum size (number of nodes)
   - Balanced structure when appropriate

## Expression Evaluation

### Evaluation Process

1. **Recursive Evaluation**:
   - Evaluate child nodes first
   - Apply operator or function to child results
   - Handle special cases and errors

2. **Vectorized Evaluation**:
   - Evaluate expression on multiple input points simultaneously
   - Optimize for computational efficiency

3. **Caching**:
   - Cache evaluation results for common subexpressions
   - Avoid redundant calculations

### Special Handling for Riemann Zeta Function

1. **Built-in Zeta Function**:
   - Implement efficient calculation of ζ(s) for complex s
   - Use series approximations and asymptotic formulas

2. **Zero Detection**:
   - Implement methods to detect zeros of expressions
   - Focus on the critical strip (0 < Re(s) < 1)

3. **Critical Line Testing**:
   - Specialized methods to test behavior on the critical line (Re(s) = 1/2)
   - Evaluate expressions at known zeros of the zeta function

## Expression Manipulation

### Simplification

1. **Algebraic Simplification**:
   - Combine like terms
   - Apply basic identities (e.g., sin²(x) + cos²(x) = 1)
   - Remove redundant operations

2. **Numerical Simplification**:
   - Evaluate constant subexpressions
   - Approximate complex constants when appropriate

### Serialization and Deserialization

1. **String Representation**:
   - Convert expression trees to readable mathematical notation
   - Support LaTeX output for documentation

2. **Parsing**:
   - Parse string representations back into expression trees
   - Handle standard mathematical notation

## Implementation Plan

1. **Core Classes**:
   - `Node`: Base class for expression tree nodes
   - `Operator`, `Function`, `Variable`, `Constant`: Specialized node types
   - `ExpressionTree`: Container for the entire expression

2. **Expression Generators**:
   - `RandomExpressionGenerator`: Create random valid expressions
   - `TargetedExpressionGenerator`: Create expressions with specific properties

3. **Evaluators**:
   - `ExpressionEvaluator`: Evaluate expressions on given inputs
   - `ZetaFunctionEvaluator`: Specialized evaluator for zeta function calculations

4. **Manipulators**:
   - `ExpressionSimplifier`: Simplify expressions
   - `ExpressionMutator`: Apply mutations to expressions
   - `ExpressionCrossover`: Perform crossover between expressions

This design provides a flexible and powerful representation for mathematical expressions that can evolve to explore potential approaches to the Riemann Hypothesis.
