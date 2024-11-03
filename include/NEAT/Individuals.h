#include "NEAT/NN.h"
#include <algorithm>
#include <pain.h>

struct NeatConfig {
  std::size_t m_population_size;
  int m_num_inputs;
  int m_num_outputs;
  double m_init_mean, m_init_stdev, m_min, m_max;
  double m_mutation_rate, m_mutation_power;
  double m_replace_rate;
  pain::RNG m_rng;
  // delta formula
  double m_c1, m_c2, m_c3, dThreshold;
};

class Individual
{
public:
  Individual(Genome genome, double fitness, const NeatConfig &config)
      : m_config(config), m_genome{std::move(genome)}, m_fitness(fitness){};

  double clamp(double x) const // or "clip" is also a possible name
  {
    return std::min(m_config.m_max, std::max(m_config.m_min, x));
  }
  double replaceValue() { return clamp(m_config.m_rng.gaussian<double>()); }
  double mutateDelta(double value)
  {
    double delta =
        clamp(m_config.m_rng.gaussian<double>(0.0, m_config.m_mutation_power));
    return clamp(value + delta);
  }
  // Structural Mutations
  void mutateAddNeuron();
  void mutateAddLink();
  void mutateRemoveNeuron();
  void mutateRemoveLink();
  // Crossover
  Individual &crossover(const Individual &other) const;
  // Delta Formula δ
  double calculateDelta(const Individual &other) const;

private:
  const NeatConfig &m_config;
  Genome m_genome;
  double m_fitness;
  // delta formula:
  double m_dN;
};
