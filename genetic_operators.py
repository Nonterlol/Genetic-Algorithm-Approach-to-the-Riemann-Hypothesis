"""
Genetic Operators Implementation for Riemann Hypothesis Genetic Algorithm

This module implements the genetic operators (selection, crossover, mutation)
for evolving mathematical expressions as designed in the genetic_algorithm_design.md document.
"""

import random
import copy
from typing import List, Tuple, Callable, Dict, Any, Optional

from expression_tree import (
    ExpressionTree, Node, NodeType, OperatorNode, FunctionNode, 
    VariableNode, ConstantNode, ExpressionGenerator
)

class GeneticOperators:
    """Class implementing genetic operators for expression evolution."""
    
    def __init__(self, 
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 3,
                 elitism_count: int = 2,
                 max_depth: int = 10,
                 max_size: int = 100):
        """
        Initialize the genetic operators.
        
        Args:
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Number of individuals in tournament selection
            elitism_count: Number of best individuals to preserve unchanged
            max_depth: Maximum depth of expression trees
            max_size: Maximum number of nodes in expression trees
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.max_depth = max_depth
        self.max_size = max_size
        
        # Expression generator for creating new expressions
        self.generator = ExpressionGenerator(max_depth=3)
    
    def selection(self, population: List[ExpressionTree], 
                 fitness_values: List[float]) -> ExpressionTree:
        """
        Select an individual from the population using tournament selection.
        
        Args:
            population: List of expression trees
            fitness_values: List of fitness values corresponding to the population
            
        Returns:
            Selected expression tree
        """
        # Create a list of (expression, fitness) tuples
        population_with_fitness = list(zip(population, fitness_values))
        
        # Select tournament_size individuals randomly
        tournament = random.sample(population_with_fitness, 
                                  min(self.tournament_size, len(population_with_fitness)))
        
        # Return the individual with the highest fitness
        return max(tournament, key=lambda x: x[1])[0]
    
    def crossover(self, parent1: ExpressionTree, parent2: ExpressionTree) -> Tuple[ExpressionTree, ExpressionTree]:
        """
        Perform crossover between two parent expressions.
        
        Args:
            parent1: First parent expression tree
            parent2: Second parent expression tree
            
        Returns:
            Tuple of two offspring expression trees
        """
        # Create deep copies of parents to avoid modifying the originals
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        
        # If either parent is empty, return copies of the parents
        if not offspring1.root or not offspring2.root:
            return offspring1, offspring2
        
        # Get all nodes from both parents
        nodes1 = offspring1.get_all_nodes()
        nodes2 = offspring2.get_all_nodes()
        
        # Skip the root nodes to avoid creating invalid expressions
        nodes1 = nodes1[1:] if len(nodes1) > 1 else []
        nodes2 = nodes2[1:] if len(nodes2) > 1 else []
        
        # If either list is empty, return copies of the parents
        if not nodes1 or not nodes2:
            return offspring1, offspring2
        
        # Select random crossover points
        crossover_point1 = random.choice(nodes1)
        crossover_point2 = random.choice(nodes2)
        
        # Find paths to the selected nodes
        path1 = self._find_path(offspring1.root, crossover_point1)
        path2 = self._find_path(offspring2.root, crossover_point2)
        
        if not path1 or not path2:
            # Paths not found, return copies of the parents
            return offspring1, offspring2
        
        # Get the subtrees at the crossover points
        subtree1 = self._get_subtree_at_path(offspring1.root, path1)
        subtree2 = self._get_subtree_at_path(offspring2.root, path2)
        
        if not subtree1 or not subtree2:
            # Subtrees not found, return copies of the parents
            return offspring1, offspring2
        
        # Create copies of the subtrees
        subtree1_copy = subtree1.copy()
        subtree2_copy = subtree2.copy()
        
        # Swap the subtrees
        self._replace_subtree_at_path(offspring1.root, path1, subtree2_copy)
        self._replace_subtree_at_path(offspring2.root, path2, subtree1_copy)
        
        # Check if the resulting trees are too deep or too large
        if (offspring1.depth() > self.max_depth or 
            offspring1.count_nodes() > self.max_size):
            offspring1 = parent1.copy()
        
        if (offspring2.depth() > self.max_depth or 
            offspring2.count_nodes() > self.max_size):
            offspring2 = parent2.copy()
        
        return offspring1, offspring2
    
    def _find_path(self, root: Node, target: Node) -> List[int]:
        """
        Find the path from the root to the target node.
        
        Args:
            root: Root node of the tree
            target: Target node to find
            
        Returns:
            List of child indices representing the path, or empty list if not found
        """
        if root is target:
            return []
        
        for i, child in enumerate(root.children):
            path = self._find_path(child, target)
            if path is not None:
                return [i] + path
        
        return None
    
    def _get_subtree_at_path(self, root: Node, path: List[int]) -> Optional[Node]:
        """
        Get the subtree at the specified path.
        
        Args:
            root: Root node of the tree
            path: List of child indices representing the path
            
        Returns:
            The node at the specified path, or None if the path is invalid
        """
        if not path:
            return root
        
        if path[0] >= len(root.children):
            return None
        
        return self._get_subtree_at_path(root.children[path[0]], path[1:])
    
    def _replace_subtree_at_path(self, root: Node, path: List[int], new_subtree: Node) -> bool:
        """
        Replace the subtree at the specified path.
        
        Args:
            root: Root node of the tree
            path: List of child indices representing the path
            new_subtree: The new subtree to insert
            
        Returns:
            True if the replacement was successful, False otherwise
        """
        if not path:
            return False  # Cannot replace the root this way
        
        if len(path) == 1:
            if path[0] >= len(root.children):
                return False
            root.children[path[0]] = new_subtree
            return True
        
        if path[0] >= len(root.children):
            return False
        
        return self._replace_subtree_at_path(root.children[path[0]], path[1:], new_subtree)
    
    def mutation(self, individual: ExpressionTree) -> ExpressionTree:
        """
        Mutate an individual expression tree.
        
        Args:
            individual: Expression tree to mutate
            
        Returns:
            Mutated expression tree
        """
        # Create a deep copy to avoid modifying the original
        mutated = individual.copy()
        
        if not mutated.root:
            return mutated
        
        # Get all nodes
        nodes = mutated.get_all_nodes()
        
        # Choose a random node to mutate
        node_to_mutate = random.choice(nodes)
        
        # Apply a random mutation operator
        mutation_operators = [
            self._point_mutation,
            self._subtree_mutation,
            self._insert_function,
            self._constant_perturbation
        ]
        
        mutation_operator = random.choice(mutation_operators)
        success = mutation_operator(mutated, node_to_mutate)
        
        # If mutation failed or resulted in an invalid tree, return the original
        if not success or mutated.depth() > self.max_depth or mutated.count_nodes() > self.max_size:
            return individual.copy()
        
        return mutated
    
    def _point_mutation(self, tree: ExpressionTree, node: Node) -> bool:
        """
        Replace a node with another of the same type.
        
        Args:
            tree: Expression tree containing the node
            node: Node to mutate
            
        Returns:
            True if mutation was successful, False otherwise
        """
        if node.type == NodeType.OPERATOR:
            # Replace with another operator of the same arity
            current_arity = len(node.children)
            compatible_operators = [op for op, info in OperatorNode.OPERATORS.items() 
                                   if info['arity'] == current_arity]
            
            if not compatible_operators:
                return False
            
            new_operator = random.choice(compatible_operators)
            if new_operator != node.value:
                node.value = new_operator
                return True
        
        elif node.type == NodeType.FUNCTION:
            # Replace with another function
            new_function = random.choice(list(FunctionNode.FUNCTIONS.keys()))
            if new_function != node.value:
                node.value = new_function
                return True
        
        elif node.type == NodeType.VARIABLE:
            # Replace with another variable
            variables = ['s', 't', 'z']
            new_variable = random.choice(variables)
            if new_variable != node.value:
                node.value = new_variable
                return True
        
        elif node.type == NodeType.CONSTANT:
            # Replace with another constant
            if random.random() < 0.5:
                # Use a special constant
                special_constants = list(ConstantNode.SPECIAL_CONSTANTS.keys())
                new_constant = random.choice(special_constants)
                node.value = ConstantNode.SPECIAL_CONSTANTS[new_constant]
                node.name = new_constant
            else:
                # Generate a random constant
                if random.random() < 0.7:
                    # Real number
                    node.value = random.uniform(-10, 10)
                else:
                    # Complex number
                    node.value = complex(random.uniform(-5, 5), random.uniform(-5, 5))
                node.name = None
            return True
        
        return False
    
    def _subtree_mutation(self, tree: ExpressionTree, node: Node) -> bool:
        """
        Replace a subtree with a randomly generated one.
        
        Args:
            tree: Expression tree containing the node
            node: Root of the subtree to replace
            
        Returns:
            True if mutation was successful, False otherwise
        """
        # Find the path to the node
        path = self._find_path(tree.root, node)
        
        if path is None:
            return False
        
        # Generate a new random subtree
        new_subtree = self.generator.generate_random_expression(depth=0)
        
        # Replace the subtree
        if not path:  # Root node
            tree.root = new_subtree
        else:
            success = self._replace_subtree_at_path(tree.root, path, new_subtree)
            if not success:
                return False
        
        return True
    
    def _insert_function(self, tree: ExpressionTree, node: Node) -> bool:
        """
        Insert a function node above an existing subtree.
        
        Args:
            tree: Expression tree containing the node
            node: Root of the subtree to wrap with a function
            
        Returns:
            True if mutation was successful, False otherwise
        """
        # Find the path to the node
        path = self._find_path(tree.root, node)
        
        if path is None:
            return False
        
        # Create a new function node
        function_name = random.choice(list(FunctionNode.FUNCTIONS.keys()))
        new_function = FunctionNode(function_name)
        
        # Make the existing node a child of the new function
        new_function.add_child(node.copy())
        
        # Replace the existing node with the new function
        if not path:  # Root node
            tree.root = new_function
        else:
            success = self._replace_subtree_at_path(tree.root, path, new_function)
            if not success:
                return False
        
        return True
    
    def _constant_perturbation(self, tree: ExpressionTree, node: Node) -> bool:
        """
        Perturb a constant value slightly.
        
        Args:
            tree: Expression tree containing the node
            node: Constant node to perturb
            
        Returns:
            True if mutation was successful, False otherwise
        """
        if node.type != NodeType.CONSTANT:
            return False
        
        if node.name:  # Special constant, don't perturb
            return False
        
        if isinstance(node.value, complex):
            # Perturb complex number
            perturbation_real = random.uniform(-0.5, 0.5)
            perturbation_imag = random.uniform(-0.5, 0.5)
            node.value = complex(node.value.real + perturbation_real,
                                node.value.imag + perturbation_imag)
        else:
            # Perturb real number
            perturbation = random.uniform(-0.5, 0.5)
            node.value += perturbation
        
        return True
    
    def simplify_expression(self, expression: ExpressionTree) -> ExpressionTree:
        """
        Apply algebraic simplification to an expression tree.
        
        Args:
            expression: Expression tree to simplify
            
        Returns:
            Simplified expression tree
        """
        # This is a placeholder implementation
        # In a real implementation, I would apply various simplification rules
        
        # For now, just return a copy of the expression
        return expression.copy()
    
    def evolve_population(self, population: List[ExpressionTree], 
                         fitness_function: Callable[[ExpressionTree, List[ExpressionTree]], float]) -> List[ExpressionTree]:
        """
        Evolve a population of expression trees using selection, crossover, and mutation.
        
        Args:
            population: List of expression trees
            fitness_function: Function to evaluate fitness of expressions
            
        Returns:
            New population of expression trees
        """
        # Calculate fitness for each individual
        fitness_values = [fitness_function(ind, population) for ind in population]
        
        # Create a list of (expression, fitness) tuples and sort by fitness
        population_with_fitness = list(zip(population, fitness_values))
        population_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Apply elitism: keep the best individuals unchanged
        new_population = [ind.copy() for ind, _ in population_with_fitness[:self.elitism_count]]
        
        # Fill the rest of the population with offspring
        while len(new_population) < len(population):
            
(Content truncated due to size limit. Use line ranges to read in chunks)