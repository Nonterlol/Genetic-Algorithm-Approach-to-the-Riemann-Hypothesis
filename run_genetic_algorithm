"""
Main script to execute the genetic algorithm for the Riemann Hypothesis

This script runs the genetic algorithm with various parameter configurations
and monitors the progress to find expressions that might provide insights
into the Riemann Hypothesis.
"""

import os
import sys
import time
import argparse
from datetime import datetime

from expression_tree import ExpressionTree, ExpressionGenerator
from fitness_function import RiemannFitness
from genetic_operators import GeneticOperators
from evolution_engine import EvolutionEngine

def create_results_directory(base_dir="../results"):
    """Create a timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def run_with_config(config, results_dir):
    """Run the genetic algorithm with a specific configuration."""
    # Create a subdirectory for this configuration
    config_name = f"pop{config['population_size']}_gen{config['max_generations']}_mut{int(config['mutation_rate']*100)}_cross{int(config['crossover_rate']*100)}"
    config_dir = os.path.join(results_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config_dir, "config.txt"), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Create and run the evolution engine
    engine = EvolutionEngine(
        population_size=config['population_size'],
        max_generations=config['max_generations'],
        mutation_rate=config['mutation_rate'],
        crossover_rate=config['crossover_rate'],
        tournament_size=config['tournament_size'],
        elitism_count=config['elitism_count'],
        max_depth=config['max_depth'],
        max_size=config['max_size'],
        num_test_zeros=config['num_test_zeros'],
        num_validation_zeros=config['num_validation_zeros'],
        results_dir=config_dir
    )
    
    # Run the evolution
    engine.run_evolution(checkpoint_interval=config['checkpoint_interval'])
    
    # Analyze the results
    analysis = engine.analyze_results()
    
    return analysis

def main():
    """Main function to run the genetic algorithm with various configurations."""
    parser = argparse.ArgumentParser(description='Run the Riemann Hypothesis Genetic Algorithm')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with minimal parameters')
    parser.add_argument('--medium', action='store_true', help='Run a medium-sized experiment')
    parser.add_argument('--full', action='store_true', help='Run a full-scale experiment')
    args = parser.parse_args()
    
    # Create results directory
    results_dir = create_results_directory()
    print(f"Results will be saved to: {results_dir}")
    
    # Define configurations
    if args.quick:
        configs = [
            {
                'name': 'Quick Test',
                'population_size': 20,
                'max_generations': 10,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'tournament_size': 3,
                'elitism_count': 2,
                'max_depth': 5,
                'max_size': 50,
                'num_test_zeros': 5,
                'num_validation_zeros': 2,
                'checkpoint_interval': 2
            }
        ]
    elif args.medium:
        configs = [
            {
                'name': 'Medium Run - Balanced',
                'population_size': 100,
                'max_generations': 50,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'tournament_size': 5,
                'elitism_count': 5,
                'max_depth': 8,
                'max_size': 100,
                'num_test_zeros': 20,
                'num_validation_zeros': 10,
                'checkpoint_interval': 5
            },
            {
                'name': 'Medium Run - High Mutation',
                'population_size': 100,
                'max_generations': 50,
                'mutation_rate': 0.5,
                'crossover_rate': 0.5,
                'tournament_size': 5,
                'elitism_count': 5,
                'max_depth': 8,
                'max_size': 100,
                'num_test_zeros': 20,
                'num_validation_zeros': 10,
                'checkpoint_interval': 5
            }
        ]
    elif args.full:
        configs = [
            {
                'name': 'Full Run - Balanced',
                'population_size': 500,
                'max_generations': 200,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'tournament_size': 7,
                'elitism_count': 10,
                'max_depth': 10,
                'max_size': 200,
                'num_test_zeros': 100,
                'num_validation_zeros': 20,
                'checkpoint_interval': 10
            },
            {
                'name': 'Full Run - High Mutation',
                'population_size': 500,
                'max_generations': 200,
                'mutation_rate': 0.5,
                'crossover_rate': 0.5,
                'tournament_size': 7,
                'elitism_count': 10,
                'max_depth': 10,
                'max_size': 200,
                'num_test_zeros': 100,
                'num_validation_zeros': 20,
                'checkpoint_interval': 10
            },
            {
                'name': 'Full Run - Deep Trees',
                'population_size': 500,
                'max_generations': 200,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'tournament_size': 7,
                'elitism_count': 10,
                'max_depth': 15,
                'max_size': 300,
                'num_test_zeros': 100,
                'num_validation_zeros': 20,
                'checkpoint_interval': 10
            }
        ]
    else:
        # Default to a small test run
        configs = [
            {
                'name': 'Default Test',
                'population_size': 50,
                'max_generations': 20,
                'mutation_rate': 0.3,
                'crossover_rate': 0.7,
                'tournament_size': 3,
                'elitism_count': 2,
                'max_depth': 6,
                'max_size': 80,
                'num_test_zeros': 10,
                'num_validation_zeros': 5,
                'checkpoint_interval': 5
            }
        ]
    
    # Run each configuration
    results = []
    for config in configs:
        print(f"\n\n{'='*80}")
        print(f"Running configuration: {config['name']}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        analysis = run_with_config(config, results_dir)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"\nCompleted in {elapsed_time:.2f} seconds")
        print(f"Best fitness: {analysis['best_fitness']:.6f}")
        print(f"Best individual: {analysis['best_individual']}")
        
        results.append({
            'config': config,
            'analysis': analysis,
            'elapsed_time': elapsed_time
        })
    
    # Summarize all results
    with open(os.path.join(results_dir, "summary.txt"), 'w') as f:
        f.write("Riemann Hypothesis Genetic Algorithm - Results Summary\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            config = result['config']
            analysis = result['analysis']
            
            f.write(f"Configuration: {config['name']}\n")
            f.write(f"Population Size: {config['population_size']}\n")
            f.write(f"Max Generations: {config['max_generations']}\n")
            f.write(f"Mutation Rate: {config['mutation_rate']}\n")
            f.write(f"Crossover Rate: {config['crossover_rate']}\n")
            f.write(f"Best Fitness: {analysis['best_fitness']:.6f}\n")
            f.write(f"Best Individual: {analysis['best_individual']}\n")
            f.write(f"Complexity: {analysis['complexity']} nodes\n")
            f.write(f"Depth: {analysis['depth']}\n")
            f.write(f"Elapsed Time: {result['elapsed_time']:.2f} seconds\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\nAll runs completed. Results saved to {results_dir}")
    
    # Return the best result
    best_result = max(results, key=lambda x: x['analysis']['best_fitness'])
    print(f"\nBest overall result:")
    print(f"Configuration: {best_result['config']['name']}")
    print(f"Best fitness: {best_result['analysis']['best_fitness']:.6f}")
    print(f"Best individual: {best_result['analysis']['best_individual']}")
    
    return best_result

if __name__ == "__main__":
    main()
