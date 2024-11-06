#pragma once

#include "Core.h"
#include "NEAT/NN.h"
#include <algorithm>
#include <pain.h>

struct NeatConfig {
  int m_populationSize;
  int m_numInputs;
  int m_numOutputs;
  // non-structural mutation
  double m_initMean, m_initStdev, m_min, m_max;
  double m_mutationRate;
  double m_mutationPower;
  double m_biasMutationRate;
  double m_replacementRate;
  double m_biasReplacementRate;
  // delta formula
  double m_c1, m_c2, m_c3, dThreshold;
  // structural mutations
  double m_probAddNode;
  double m_probAddConn;
  double m_probRmNode;
  double m_probRmConn;
};

class Individual
{
public:
  int m_speciesID = -1;
  double m_fitness = 0.0;
  Individual(Genome genome, const NeatConfig &config, const pain::RNG &rng)
      : m_config{config}, m_rng{rng}, m_genome(std::move(genome)){};

  bool fit(const std::vector<double> &inputs);
  double clamp(double x) const // or "clip" is also a possible name
  {
    return std::min(m_config.m_max, std::max(m_config.m_min, x));
  }
  double replaceValue() { return clamp(m_rng.gaussian<double>()); }
  double mutateDelta(double value)
  {
    double delta = clamp(m_rng.gaussian<double>(0.0, m_config.m_mutationPower));
    return clamp(value + delta);
  }
  // Structural Mutations
  void mutateAddNeuron();
  void mutateAddLink();
  void mutateRemoveNeuron();
  void mutateRemoveLink();
  void nonStructuralMutate(); // define this
  // Crossover
  Individual crossover(const Individual &other,
                       std::vector<InnovationStatic> &populationInnovs) const;
  // Delta Formula δ
  double calculateDelta(const Individual &other) const;

  Individual clone() const { return Individual(m_genome, m_config, m_rng); }
  ~Individual() = default;
  Individual(Individual &&o)
      : m_speciesID(o.m_speciesID), m_fitness(o.m_fitness),
        m_config(o.m_config), m_rng(o.m_rng), m_genome(std::move(o.m_genome))
  {
  }
  Individual &operator=(Individual &&o)
  {
    if (this != &o) {
      m_speciesID = o.m_speciesID;      // Move m_speciesID
      m_fitness = o.m_fitness;          // Move m_fitness
      m_genome = std::move(o.m_genome); // Move m_genome
    }
    return *this;
  }
  NONCOPYABLE(Individual)

private:
  const NeatConfig &m_config;
  const pain::RNG &m_rng;
  Genome m_genome;
  // delta formula:
  double m_dN = 1.0;
};
