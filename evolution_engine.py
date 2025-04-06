"""
Evolution Loop and Visualization for Riemann Hypothesis Genetic Algorithm

This module implements the main evolution loop and visualization components
for the genetic algorithm approach to the Riemann Hypothesis.
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from expression_tree import ExpressionTree, ExpressionGenerator
from fitness_function import RiemannFitness
from genetic_operators import GeneticOperators

class EvolutionEngine:
    """Main engine for running the genetic algorithm evolution."""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 1000,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 5,
                 elitism_count: int = 2,
                 max_depth: int = 10,
                 max_size: int = 100,
                 num_test_zeros: int = 100,
                 num_validation_zeros: int = 20,
                 results_dir: str = "../results"):
        """
        Initialize the evolution engine.
        
        Args:
            population_size: Number of individuals in the population
            max_generations: Maximum number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Number of individuals in tournament selection
            elitism_count: Number of best individuals to preserve unchanged
            max_depth: Maximum depth of expression trees
            max_size: Maximum number of nodes in expression trees
            num_test_zeros: Number of known zeta zeros to test against
            num_validation_zeros: Number of zeros to reserve for validation
            results_dir: Directory to save results
        """
        self.population_size = population_size
        self.max_generations = max_generations
        
        # Initialize components
        self.generator = ExpressionGenerator(max_depth=5)
        self.fitness_evaluator = RiemannFitness(
            num_test_zeros=num_test_zeros,
            num_validation_zeros=num_validation_zeros
        )
        self.genetic_ops = GeneticOperators(
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            tournament_size=tournament_size,
            elitism_count=elitism_count,
            max_depth=max_depth,
            max_size=max_size
        )
        
        # Create results directory
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.current_generation = 0
        self.population = []
        self.fitness_history = []
        self.best_individual_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        
        # Logging
        self.log_file = os.path.join(self.results_dir, "evolution_log.txt")
        
    def initialize_population(self) -> None:
        """Initialize the population with random expressions."""
        self.population = [
            self.generator.generate_full_expression_tree() 
            for _ in range(self.population_size)
        ]
        
        # Add some special individuals that might be promising
        if len(self.population) > 5:
            # Replace a few random individuals with special ones
            special_individuals = self._create_special_individuals()
            for i, ind in enumerate(special_individuals):
                if i < len(self.population):
                    self.population[i] = ind
    
    def _create_special_individuals(self) -> List[ExpressionTree]:
        """Create special individuals that might be promising starting points."""
        special_individuals = []
        
        # Import all necessary classes
        from expression_tree import OperatorNode, VariableNode, ConstantNode, FunctionNode, ExpressionTree
        
        # Individual 1: s - 0.5 (tests if s is on the critical line)
        root = OperatorNode('-')
        root.add_child(VariableNode('s'))
        root.add_child(ConstantNode(0.5))
        special_individuals.append(ExpressionTree(root))
        
        # Individual 2: Re(s) - 0.5 (explicitly tests real part)
        root = OperatorNode('-')
        re_node = FunctionNode('re')
        re_node.add_child(VariableNode('s'))
        root.add_child(re_node)
        root.add_child(ConstantNode(0.5))
        special_individuals.append(ExpressionTree(root))
        
        # Individual 3: zeta(s) (direct zeta function)
        root = FunctionNode('zeta')
        root.add_child(VariableNode('s'))
        special_individuals.append(ExpressionTree(root))
        
        # Individual 4: abs(zeta(s)) (magnitude of zeta function)
        root = FunctionNode('abs')
        zeta_node = FunctionNode('zeta')
        zeta_node.add_child(VariableNode('s'))
        root.add_child(zeta_node)
        special_individuals.append(ExpressionTree(root))
        
        # Individual 5: sin(pi * s / 2) (related to functional equation)
        root = FunctionNode('sin')
        mult_node = OperatorNode('*')
        div_node = OperatorNode('/')
        mult_node.add_child(ConstantNode('pi'))
        mult_node.add_child(div_node)
        div_node.add_child(VariableNode('s'))
        div_node.add_child(ConstantNode(2))
        root.add_child(mult_node)
        special_individuals.append(ExpressionTree(root))
        
        return special_individuals
    
    def evaluate_population(self) -> List[float]:
        """
        Evaluate the fitness of all individuals in the population.
        
        Returns:
            List of fitness values
        """
        fitness_values = []
        for individual in self.population:
            fitness = self.fitness_evaluator.calculate_fitness(individual, self.population)
            fitness_values.append(fitness)
        
        return fitness_values
    
    def run_evolution(self, checkpoint_interval: int = 10) -> None:
        """
        Run the evolution process.
        
        Args:
            checkpoint_interval: Number of generations between checkpoints
        """
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        # Start evolution
        self.log(f"Starting evolution with population size {self.population_size}")
        self.log(f"Maximum generations: {self.max_generations}")
        
        start_time = time.time()
        
        for generation in range(self.current_generation, self.max_generations):
            self.current_generation = generation
            
            # Update weights in fitness evaluator based on generation
            self.fitness_evaluator.update_weights(generation, self.max_generations)
            
            # Evaluate population
            fitness_values = self.evaluate_population()
            self.fitness_history.append(fitness_values)
            
            # Track statistics
            best_idx = np.argmax(fitness_values)
            best_individual = self.population[best_idx]
            best_fitness = fitness_values[best_idx]
            avg_fitness = np.mean(fitness_values)
            diversity = self._calculate_diversity()
            
            self.best_individual_history.append(best_individual.copy())
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # Log progress
            self.log(f"Generation {generation}: Best fitness = {best_fitness:.4f}, Avg fitness = {avg_fitness:.4f}, Diversity = {diversity:.4f}")
            self.log(f"Best individual: {best_individual.to_string()}")
            
            # Check for termination conditions
            if self._check_termination(best_fitness):
                self.log("Termination condition met. Stopping evolution.")
                break
            
            # Evolve population
            self.population = self.genetic_ops.evolve_population(
                self.population,
                lambda ind, pop: self.fitness_evaluator.calculate_fitness(ind, pop)
            )
            
            # Save checkpoint periodically
            if generation % checkpoint_interval == 0:
                self.save_checkpoint()
                self.visualize_progress()
        
        # Final checkpoint and visualization
        self.save_checkpoint()
        self.visualize_progress()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        self.log(f"Evolution completed in {elapsed_time:.2f} seconds")
        self.log(f"Final best fitness: {self.best_fitness_history[-1]:.4f}")
        self.log(f"Final best individual: {self.best_individual_history[-1].to_string()}")
    
    def _calculate_diversity(self) -> float:
        """
        Calculate the diversity of the population.
        
        Returns:
            Diversity measure between 0 and 1
        """
        if not self.population or len(self.population) < 2:
            return 0.0
        
        # Sample pairs of individuals and calculate average distance
        num_samples = min(100, len(self.population) * (len(self.population) - 1) // 2)
        distances = []
        
        for _ in range(num_samples):
            i, j = random.sample(range(len(self.population)), 2)
            ind1 = self.population[i]
            ind2 = self.population[j]
            distance = self.fitness_evaluator._calculate_expression_distance(ind1, ind2)
            distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _check_termination(self, best_fitness: float) -> bool:
        """
        Check if termination conditions are met.
        
        Args:
            best_fitness: Current best fitness
            
        Returns:
            True if termination conditions are met, False otherwise
        """
        # Check if we've reached a very high fitness
        if best_fitness > 0.95:
            return True
        
        # Check for fitness plateau
        if (len(self.best_fitness_history) > 50 and
            abs(self.best_fitness_history[-1] - self.best_fitness_history[-50]) < 0.001):
            return True
        
        return False
    
    def save_checkpoint(self) -> None:
        """Save a checkpoint of the current state."""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save best individual
        best_idx = np.argmax(self.fitness_history[-1]) if self.fitness_history else 0
        best_individual = self.population[best_idx] if self.population else None
        
        if best_individual:
            best_file = os.path.join(checkpoint_dir, f"best_gen_{self.current_generation}.txt")
            with open(best_file, 'w') as f:
                f.write(best_individual.to_string())
                f.write(f"\nFitness: {self.fitness_history[-1][best_idx]:.6f}")
        
        # Save statistics
        stats_file = os.path.join(checkpoint_dir, f"stats_gen_{self.current_generation}.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Generation: {self.current_generation}\n")
            f.write(f"Best fitness: {max(self.fitness_history[-1]) if self.fitness_history else 0:.6f}\n")
            f.write(f"Average fitness: {np.mean(self.fitness_history[-1]) if self.fitness_history else 0:.6f}\n")
            f.write(f"Diversity: {self.diversity_history[-1] if self.diversity_history else 0:.6f}\n")
            
            # Save top 5 individuals
            if self.population and self.fitness_history:
                f.write("\nTop 5 individuals:\n")
                indices = np.argsort(self.fitness_history[-1])[::-1][:5]
                for i, idx in enumerate(indices):
                    f.write(f"{i+1}. {self.population[idx].to_string()}\n")
                    f.write(f"   Fitness: {self.fitness_history[-1][idx]:.6f}\n")
    
    def visualize_progress(self) -> None:
        """Generate visualizations of the evolution progress."""
        if not self.best_fitness_history:
            return
        
        viz_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot fitness over generations
        plt.figure(figsize=(12, 6))
        generations = list(range(len(self.best_fitness_history)))
        
        plt.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(generations, self.avg_fitness_history, 'r-', label='Average Fitness')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(viz_dir, f"fitness_gen_{self.current_generation}.png"))
        plt.close()
        
        # Plot diversity over generations
        plt.figure(figsize=(12, 6))
        plt.plot(generations, self.diversity_history, 'g-', label='Population Diversity')
        
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.title('Population Diversity')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(viz_dir, f"diversity_gen_{self.current_generation}.png"))
        plt.close()
        
        # Plot complexity of best individual over generations
        complexity_history = [ind.count_nodes() for ind in self.best_individual_history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(generations, complexity_history, 'm-', label='Best Individual Complexity')
        
        plt.xlabel('Generation')
        plt.ylabel('Number of Nodes')
        plt.title('Best Individual Complexity')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(viz_dir, f"complexity_gen_{self.current_generation}.png"))
        plt.close()
    
    def log(self, message: str) -> None:
        """
        Log a message to the log file and print it.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the results of the evolution.
        
        Returns:
            Dictionary of analysis results
        """
        if not self.best_individual_history:
            return {"error": "No evolution history available"}
        
        # Get the best individual from the last generation
        best_individual = self.best_individual_history[-1]
        best_fitness = self.best_fitness_history[-1]
        
        # Analyze the best individual
        analysis = {
            "best_individual": best_individual.to_string(),
            "best_fitness": best_fitness,
            "complexity": best_individual.count_nodes(),
            "depth": best_individual.depth(),
            "generations_run": len(self.best_fitness_history),
            "fitness_improvement": self.best_fitness_history[-1] - self.best_fitness_history[0] if len(self.best_fitness_history) > 1 else 0,
            "plateau_generations": self._count_plateau_generations(),
        }
        
        # Save the analysis
        analysis_file = os.path.join(self.results_dir, "final_analysis.txt")
        with open(analysis_file, 'w') as f:
            f.write("Final Analysis\n")
            f.write("=============\n\n")
         
(Content truncated due to size limit. Use line ranges to read in chunks)